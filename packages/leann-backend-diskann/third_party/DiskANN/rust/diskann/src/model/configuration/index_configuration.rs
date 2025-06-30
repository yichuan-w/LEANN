/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Index configuration.

use vector::Metric;

use super::index_write_parameters::IndexWriteParameters;

/// The index configuration
#[derive(Debug, Clone)]
pub struct IndexConfiguration {
    /// Index write parameter
    pub index_write_parameter: IndexWriteParameters,

    /// Distance metric
    pub dist_metric: Metric,

    /// Dimension of the raw data
    pub dim: usize,
    
    /// Aligned dimension - round up dim to the nearest multiple of 8
    pub aligned_dim: usize,

    /// Total number of points in given data set
    pub max_points: usize,

    /// Number of points which are used as initial candidates when iterating to
    /// closest point(s). These are not visible externally and won't be returned
    /// by search. DiskANN forces at least 1 frozen point for dynamic index.
    /// The frozen points have consecutive locations.
    pub num_frozen_pts: usize,

    /// Calculate distance by PQ or not
    pub use_pq_dist: bool,

    /// Number of PQ chunks
    pub num_pq_chunks: usize,

    /// Use optimized product quantization
    /// Currently not supported
    pub use_opq: bool,

    /// potential for growth. 1.2 means the index can grow by up to 20%.
    pub growth_potential: f32,

    // TODO: below settings are not supported in current iteration
    // pub concurrent_consolidate: bool,
    // pub has_built: bool,
    // pub save_as_one_file: bool,
    // pub dynamic_index: bool,
    // pub enable_tags: bool,
    // pub normalize_vecs: bool,
}

impl IndexConfiguration {
    /// Create IndexConfiguration instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: Metric, 
        dim: usize, 
        aligned_dim: usize,
        max_points: usize, 
        use_pq_dist: bool, 
        num_pq_chunks: usize, 
        use_opq: bool, 
        num_frozen_pts: usize, 
        growth_potential: f32, 
        index_write_parameter: IndexWriteParameters
    ) -> Self {
        Self {
            index_write_parameter,
            dist_metric,
            dim,
            aligned_dim,
            max_points,
            num_frozen_pts,
            use_pq_dist,
            num_pq_chunks,
            use_opq,
            growth_potential,
        }
    }

    /// Get the size of adjacency list that we build out.
    pub fn write_range(&self) -> usize {
        self.index_write_parameter.max_degree as usize
    }
}
