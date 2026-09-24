//! # Foyer Cache
//!
//! This module provides an implementation of an in-memory cache using the Foyer library.
//! The cache is designed to store and retrieve cached blocks, indexes, and filters
//! associated with SSTable IDs.
//!
//! ## Features
//!
//! - **Asynchronous Operations**: Utilizes Foyer's `Cache` to perform cache operations asynchronously.
//! - **Custom Weigher**: Implements a custom weigher to account for the size of cached blocks.
//! - **Flexible Configuration**: Allows customization of cache parameters such as maximum capacity.
//!
//! ## Examples
//!
//!
//! ```
//! use slatedb::{Db, Error};
//! use slatedb::db_cache::foyer::FoyerCache;
//! use slatedb::object_store::memory::InMemory;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Error> {
//!     let object_store = Arc::new(InMemory::new());
//!     let db = Db::builder("test_db", object_store)
//!         .with_db_cache(Arc::new(FoyerCache::new()))
//!         .build()
//!         .await?;
//!     Ok(())
//! }
//! ```
//!

use crate::db_cache::{CacheLoader, CachedEntry, CachedKey, DbCache, DEFAULT_MAX_CAPACITY};
use crate::error::SlateDBError;
use async_trait::async_trait;
use std::sync::{Arc, Mutex, Weak};
use sysinfo::{CpuRefreshKind, System};

/// Live caches only. Foyer's `foyer_memory_usage` gauge is not decremented
/// when a cache is cleared on drop, so a process that opens a lane, closes
/// it, and opens the next one reports a sum that walks past the cap.
struct LiveCache {
    name: String,
    usage: Weak<dyn Fn() -> u64 + Send + Sync>,
}

static LIVE_CACHES: Mutex<Vec<LiveCache>> = Mutex::new(Vec::new());

fn live_caches() -> std::sync::MutexGuard<'static, Vec<LiveCache>> {
    LIVE_CACHES.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Sum of `usage()` over caches still alive under `name`. Dropped caches
/// contribute nothing.
pub fn live_occupancy(name: &str) -> u64 {
    let mut live = live_caches();
    live.retain(|slot| slot.usage.strong_count() > 0);
    live.iter()
        .filter(|slot| slot.name == name)
        .filter_map(|slot| slot.usage.upgrade())
        .map(|usage| usage())
        .sum()
}

fn register_live(name: String, usage: &Arc<dyn Fn() -> u64 + Send + Sync>) {
    live_caches().push(LiveCache {
        name,
        usage: Arc::downgrade(usage),
    });
}

/// The options for the Foyer cache.
#[derive(Clone, Copy, Debug)]
pub struct FoyerCacheOptions {
    pub max_capacity: u64,
    pub shards: usize,
}

impl Default for FoyerCacheOptions {
    fn default() -> Self {
        Self {
            max_capacity: DEFAULT_MAX_CAPACITY,
            shards: {
                let mut sys = System::new();
                sys.refresh_cpu_specifics(CpuRefreshKind::nothing());
                sys.cpus().len()
            },
        }
    }
}

/// A cache implementation using the Foyer library.
///
/// This struct wraps a Foyer cache, providing an in-memory caching solution
/// for storing and retrieving cached blocks associated with SSTable IDs.
///
/// # Fields
///
/// * `inner` - The underlying Foyer cache instance, which maps `CachedKey`
///   keys to `CachedEntry` values.
///
/// # Notes
///
/// The cache is configured based on the provided `FoyerCacheOptions`,
/// including settings for the maximum capacity of the cache.
/// It uses a custom weigher to account for the size of cached blocks.
pub struct FoyerCache {
    inner: foyer::Cache<CachedKey, CachedEntry>,
    /// Keeps this cache in [`live_occupancy`] until drop.
    _occupancy: Option<Arc<dyn Fn() -> u64 + Send + Sync>>,
}

impl FoyerCache {
    pub fn new() -> Self {
        Self::new_with_opts(FoyerCacheOptions::default())
    }

    pub fn new_with_opts(options: FoyerCacheOptions) -> Self {
        let cache = foyer::CacheBuilder::new(options.max_capacity as _)
            .with_weighter(|_, v: &CachedEntry| v.size())
            .with_shards(options.shards)
            .build();
        Self {
            inner: cache,
            _occupancy: None,
        }
    }

    /// Same cache as [`Self::new_with_opts`], with a stable name and a shared
    /// metrics registry. Block and meta caches must use different names so
    /// hit/miss counters do not collapse. `new_with_opts` stays unmetered
    /// because this crate has no metrics recorder of its own.
    ///
    /// Also registers the cache for [`live_occupancy`]. That reading is the
    /// current weight of caches that are still open, which the usage gauge
    /// is not: foyer leaves the gauge behind when a cache is dropped.
    pub fn new_metered(
        options: FoyerCacheOptions,
        name: impl Into<String>,
        registry: mixtrics::metrics::BoxedRegistry,
    ) -> Self {
        let name = name.into();
        let cache = foyer::CacheBuilder::new(options.max_capacity as _)
            .with_name(name.clone())
            .with_metrics_registry(registry)
            .with_weighter(|_, v: &CachedEntry| v.size())
            .with_shards(options.shards)
            .build();
        let occupancy: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new({
            let cache = cache.clone();
            move || cache.usage() as u64
        });
        register_live(name, &occupancy);
        Self {
            inner: cache,
            _occupancy: Some(occupancy),
        }
    }

    /// Current weighted occupancy of this cache. At most `max_capacity`.
    pub fn usage(&self) -> u64 {
        self.inner.usage() as u64
    }
}

impl Default for FoyerCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DbCache for FoyerCache {
    async fn get_block(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_index(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_filter(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_stats(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn insert(&self, key: CachedKey, value: CachedEntry) {
        self.inner.insert(key, value);
    }

    async fn remove(&self, key: &CachedKey) {
        self.inner.remove(key);
    }

    fn entry_count(&self) -> u64 {
        // foyer cache doesn't support an entry count estimate
        0
    }

    async fn fetch_block(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_index(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_filter(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_stats(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }
}

impl FoyerCache {
    /// Use foyer's `Cache::get_or_fetch`, which deduplicates concurrent loads for the same key.
    ///
    /// Loader errors round-trip via anyhow's source chain on the foyer error. Foyer wraps them
    /// as `ErrorKind::External` (see foyer-memory's raw.rs). We don't try to recover the original
    /// `crate::Error` value: foyer's broadcast path makes one-to-one recovery impossible for
    /// concurrent waiters, so all error returns are normalized to `SlateDBError::FoyerError`
    /// with the original chained as a source.
    async fn dedup_fetch(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        let fetch = self
            .inner
            .get_or_fetch(&key, move || async move { loader().await });
        match fetch.await {
            Ok(entry) => Ok(entry.value().clone()),
            Err(err) => Err(SlateDBError::FoyerError(Arc::new(err)).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_registry() -> mixtrics::metrics::BoxedRegistry {
        Box::new(mixtrics::registry::noop::NoopMetricsRegistry)
    }

    #[test]
    fn live_occupancy_drops_when_the_cache_drops() {
        let name = format!("live-occ-{}", std::process::id());
        let opts = FoyerCacheOptions {
            max_capacity: 4096,
            shards: 1,
        };
        {
            let cache = FoyerCache::new_metered(opts, name.clone(), noop_registry());
            assert_eq!(live_occupancy(&name), cache.usage());
        }
        assert_eq!(live_occupancy(&name), 0);
    }
}
