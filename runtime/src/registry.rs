use std::collections::{HashMap, VecDeque};

use execution::{ExecutionTag, KernelLauncher, KernelMetadata};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelRegistryError {
    ExecutionMismatch {
        key: ExecutionTag,
        value: ExecutionTag,
    },
}

#[derive(Debug, Clone)]
pub struct KernelRegistry {
    config: KernelRegistryConfig,
    stats: KernelRegistryStats,
    cache: HashMap<KernelKey, KernelLauncher>,
    cache_lru: VecDeque<KernelKey>,
    main_memory: HashMap<KernelKey, KernelLauncher>,
    main_lru: VecDeque<KernelKey>,
    secondary_storage: HashMap<KernelKey, KernelMetadata>,
}

impl KernelRegistry {
    pub fn new(config: KernelRegistryConfig) -> Self {
        Self {
            config,
            stats: KernelRegistryStats::default(),
            cache: HashMap::new(),
            cache_lru: VecDeque::new(),
            main_memory: HashMap::new(),
            main_lru: VecDeque::new(),
            secondary_storage: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        key: KernelKey,
        metadata: KernelMetadata,
    ) -> Result<(), KernelRegistryError> {
        if key.execution() != metadata.tag() {
            return Err(KernelRegistryError::ExecutionMismatch {
                key: key.execution(),
                value: metadata.tag(),
            });
        }

        if self.contains(&key) {
            return Ok(());
        }

        self.secondary_storage.insert(key, metadata);
        Ok(())
    }

    pub fn contains(&self, key: &KernelKey) -> bool {
        self.cache.contains_key(key)
            || self.main_memory.contains_key(key)
            || self.secondary_storage.contains_key(key)
    }

    pub fn select(&mut self, key: &KernelKey) -> Option<KernelLauncher> {
        if let Some(launcher) = self.cache.get(key).cloned() {
            self.stats.cache_hits += 1;
            touch_lru(&mut self.cache_lru, *key);
            return Some(launcher);
        }

        if let Some(launcher) = self.main_memory.get(key).cloned() {
            self.stats.main_memory_hits += 1;
            touch_lru(&mut self.main_lru, *key);
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
        self.cache.clear();
        self.cache_lru.clear();
        self.main_memory.clear();
        self.main_lru.clear();
        self.secondary_storage.clear();
    }

    fn promote_to_cache(&mut self, key: KernelKey, launcher: KernelLauncher) {
        if self.config.cache_capacity == 0 {
            return;
        }

        if self.cache.contains_key(&key) {
            touch_lru(&mut self.cache_lru, key);
            return;
        }

        if self.cache.len() >= self.config.cache_capacity {
            self.evict_cache();
        }

        self.cache.insert(key, launcher);
        touch_lru(&mut self.cache_lru, key);
    }

    fn evict_cache(&mut self) {
        if let Some(victim_key) = self.cache_lru.pop_back() {
            self.cache.remove(&victim_key);
        }
    }

    fn promote_to_main_memory(&mut self, key: KernelKey, launcher: KernelLauncher) {
        if self.config.main_memory_capacity == 0 {
            self.secondary_storage.insert(key, launcher.to_metadata());
            return;
        }

        if self.main_memory.contains_key(&key) {
            touch_lru(&mut self.main_lru, key);
            return;
        }

        if self.main_memory.len() >= self.config.main_memory_capacity {
            self.evict_main_memory();
        }

        self.main_memory.insert(key, launcher);
        touch_lru(&mut self.main_lru, key);
    }

    fn evict_main_memory(&mut self) {
        let Some(victim_key) = self.main_lru.pop_back() else {
            return;
        };

        if let Some(launcher) = self.main_memory.remove(&victim_key) {
            self.secondary_storage
                .insert(victim_key, launcher.to_metadata());
        }

        self.cache.remove(&victim_key);
        remove_lru_key(&mut self.cache_lru, victim_key);
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new(KernelRegistryConfig::default())
    }
}

fn touch_lru(lru: &mut VecDeque<KernelKey>, key: KernelKey) {
    remove_lru_key(lru, key);
    lru.push_front(key);
}

fn remove_lru_key(lru: &mut VecDeque<KernelKey>, key: KernelKey) {
    if let Some(pos) = lru.iter().position(|candidate| *candidate == key) {
        let _ = lru.remove(pos);
    }
}
