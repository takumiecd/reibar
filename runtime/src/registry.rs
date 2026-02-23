use std::{collections::HashMap, num::NonZeroUsize};

use execution::{KernelLauncher, KernelMetadata};
use lru::LruCache;

use crate::KernelKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelRegistryConfig {
    pub cache_capacity: usize,
    pub main_memory_capacity: usize,
}

impl Default for KernelRegistryConfig {
    fn default() -> Self {
        Self {
            cache_capacity: 8,
            main_memory_capacity: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KernelRegistryStats {
    pub cache_hits: usize,
    pub main_memory_hits: usize,
    pub secondary_hits: usize,
    pub misses: usize,
}

#[derive(Debug, Clone)]
pub struct KernelRegistry {
    config: KernelRegistryConfig,
    stats: KernelRegistryStats,
    cache: Option<LruCache<KernelKey, KernelLauncher>>,
    main_memory: Option<LruCache<KernelKey, KernelLauncher>>,
    secondary_storage: HashMap<KernelKey, KernelMetadata>,
}

impl KernelRegistry {
    pub fn new(config: KernelRegistryConfig) -> Self {
        Self {
            config,
            stats: KernelRegistryStats::default(),
            cache: NonZeroUsize::new(config.cache_capacity).map(LruCache::new),
            main_memory: NonZeroUsize::new(config.main_memory_capacity).map(LruCache::new),
            secondary_storage: HashMap::new(),
        }
    }

    pub fn register(&mut self, key: KernelKey, metadata: KernelMetadata) {
        if self.contains(&key) {
            return;
        }

        self.secondary_storage.insert(key, metadata);
    }

    pub fn contains(&self, key: &KernelKey) -> bool {
        self.cache.as_ref().is_some_and(|cache| cache.contains(key))
            || self
                .main_memory
                .as_ref()
                .is_some_and(|main_memory| main_memory.contains(key))
            || self.secondary_storage.contains_key(key)
    }

    pub fn select(&mut self, key: &KernelKey) -> Option<KernelLauncher> {
        if let Some(launcher) = self.cache.as_mut().and_then(|cache| cache.get(key).cloned()) {
            self.stats.cache_hits += 1;
            return Some(launcher);
        }

        if let Some(launcher) = self
            .main_memory
            .as_mut()
            .and_then(|main_memory| main_memory.get(key).cloned())
        {
            self.stats.main_memory_hits += 1;
            self.promote_to_cache(*key, launcher.clone());
            return Some(launcher);
        }

        if let Some(metadata) = self.secondary_storage.remove(key) {
            self.stats.secondary_hits += 1;
            let launcher = metadata.into_launcher();
            self.promote_to_main_memory(*key, launcher.clone());
            self.promote_to_cache(*key, launcher.clone());
            return Some(launcher);
        }

        self.stats.misses += 1;
        None
    }

    pub fn stats(&self) -> KernelRegistryStats {
        self.stats
    }

    pub fn clear(&mut self) {
        self.stats = KernelRegistryStats::default();
        if let Some(cache) = self.cache.as_mut() {
            cache.clear();
        }
        if let Some(main_memory) = self.main_memory.as_mut() {
            main_memory.clear();
        }
        self.secondary_storage.clear();
    }

    fn promote_to_cache(&mut self, key: KernelKey, launcher: KernelLauncher) {
        let Some(cache) = self.cache.as_mut() else {
            return;
        };

        if cache.get(&key).is_some() {
            return;
        }

        if cache.len() >= self.config.cache_capacity {
            let _ = cache.pop_lru();
        }

        cache.put(key, launcher);
    }

    fn promote_to_main_memory(&mut self, key: KernelKey, launcher: KernelLauncher) {
        if self.config.main_memory_capacity == 0 {
            self.secondary_storage.insert(key, launcher.to_metadata());
            return;
        }

        if self
            .main_memory
            .as_mut()
            .and_then(|main_memory| main_memory.get(&key))
            .is_some()
        {
            return;
        }

        if self
            .main_memory
            .as_ref()
            .is_some_and(|main_memory| main_memory.len() >= self.config.main_memory_capacity)
        {
            self.evict_main_memory();
        }

        if let Some(main_memory) = self.main_memory.as_mut() {
            main_memory.put(key, launcher);
        }
    }

    fn evict_main_memory(&mut self) {
        let Some((victim_key, launcher)) = self
            .main_memory
            .as_mut()
            .and_then(|main_memory| main_memory.pop_lru())
        else {
            return;
        };

        self.secondary_storage
            .insert(victim_key, launcher.to_metadata());

        if let Some(cache) = self.cache.as_mut() {
            let _ = cache.pop(&victim_key);
        }
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new(KernelRegistryConfig::default())
    }
}
