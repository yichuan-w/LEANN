/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Parameters for disk index construction.

use crate::common::{ANNResult, ANNError};

/// Cached nodes size in GB
const SPACE_FOR_CACHED_NODES_IN_GB: f64 = 0.25;

/// Threshold for caching in GB
const THRESHOLD_FOR_CACHING_IN_GB: f64 = 1.0;

/// Parameters specific for disk index construction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DiskIndexBuildParameters {
    /// Bound on the memory footprint of the index at search time in bytes. 
    /// Once built, the index will use up only the specified RAM limit, the rest will reside on disk.
    /// This will dictate how aggressively we compress the data vectors to store in memory. 
    /// Larger will yield better performance at search time.
    search_ram_limit: f64,

    /// Limit on the memory allowed for building the index in bytes.
    index_build_ram_limit: f64,
}

impl DiskIndexBuildParameters {
    /// Create DiskIndexBuildParameters instance
    pub fn new(search_ram_limit_gb: f64, index_build_ram_limit_gb: f64) -> ANNResult<Self> {
        let param = Self { 
            search_ram_limit: Self::get_memory_budget(search_ram_limit_gb), 
            index_build_ram_limit: index_build_ram_limit_gb * 1024_f64 * 1024_f64 * 1024_f64,
        };

        if param.search_ram_limit <= 0f64 {
            return Err(ANNError::log_index_config_error("search_ram_limit".to_string(), "RAM budget should be > 0".to_string()))
        }

        if param.index_build_ram_limit <= 0f64 {
            return Err(ANNError::log_index_config_error("index_build_ram_limit".to_string(), "RAM budget should be > 0".to_string()))
        }

        Ok(param)
    }

    /// Get search_ram_limit
    pub fn search_ram_limit(&self) -> f64 {
        self.search_ram_limit
    }

    /// Get index_build_ram_limit
    pub fn index_build_ram_limit(&self) -> f64 {
        self.index_build_ram_limit
    }

    fn get_memory_budget(mut index_ram_limit_gb: f64) -> f64 {
        if index_ram_limit_gb - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB {
            // slack for space used by cached nodes
            index_ram_limit_gb -= SPACE_FOR_CACHED_NODES_IN_GB;
        }

        index_ram_limit_gb * 1024_f64 * 1024_f64 * 1024_f64
    }
}

#[cfg(test)]
mod dataset_test {
    use super::*;

    #[test]
    fn sufficient_ram_for_caching() {
        let param = DiskIndexBuildParameters::new(1.26_f64, 1.0_f64).unwrap();
        assert_eq!(param.search_ram_limit, 1.01_f64 * 1024_f64 * 1024_f64 * 1024_f64);
    }

    #[test]
    fn insufficient_ram_for_caching() {
        let param = DiskIndexBuildParameters::new(0.03_f64, 1.0_f64).unwrap();
        assert_eq!(param.search_ram_limit, 0.03_f64 * 1024_f64 * 1024_f64 * 1024_f64);
    }
}

