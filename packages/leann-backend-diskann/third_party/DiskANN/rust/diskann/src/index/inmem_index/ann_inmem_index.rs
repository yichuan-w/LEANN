/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! ANN in-memory index abstraction

use vector::FullPrecisionDistance;

use crate::model::{vertex::{DIM_128, DIM_256, DIM_104}, IndexConfiguration};
use crate::common::{ANNResult, ANNError};

use super::InmemIndex;

/// ANN inmem-index abstraction for custom <T, N>
pub trait ANNInmemIndex<T> : Sync + Send
where T : Default + Copy + Sync + Send + Into<f32>
 {
    /// Build index
    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()>;

    /// Save index
    fn save(&mut self, filename: &str) -> ANNResult<()>;

    /// Load index
    fn load(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<()>;

    /// insert index
    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()>;

    /// Search the index for K nearest neighbors of query using given L value, for benchmarking purposes
    fn search(&self, query : &[T], k_value : usize, l_value : u32, indices : &mut[u32]) -> ANNResult<u32>;

    /// Soft deletes the nodes with the ids in the given array.
    fn soft_delete(&mut self, vertex_ids_to_delete: Vec<u32>,  num_points_to_delete: usize) -> ANNResult<()>;
}

/// Create Index<T, N> based on configuration
pub fn create_inmem_index<'a, T>(config: IndexConfiguration) -> ANNResult<Box<dyn ANNInmemIndex<T> + 'a>> 
where
    T: Default + Copy + Sync + Send + Into<f32> + 'a,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
{
    match config.aligned_dim {
        DIM_104 => {
            let index = Box::new(InmemIndex::<T, DIM_104>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        },
        DIM_128 => {
            let index = Box::new(InmemIndex::<T, DIM_128>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        },
        DIM_256 => {
            let index = Box::new(InmemIndex::<T, DIM_256>::new(config)?);
            Ok(index as Box<dyn ANNInmemIndex<T>>)
        },
        _ => Err(ANNError::log_index_error(format!("Invalid dimension: {}", config.aligned_dim))),
    }
}

#[cfg(test)]
mod dataset_test {
    use vector::Metric;

    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;

    use super::*;

    #[test]
    #[should_panic(expected = "ERROR: Data file fake_file does not exist.")]
    fn create_index_test() {
        let index_write_parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .with_saturate_graph(false)
            .with_num_threads(1)
            .build();

        let config = IndexConfiguration::new(
            Metric::L2,
            128,
            256,
            1_000_000,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
        );
        let mut index = create_inmem_index::<f32>(config).unwrap();
        index.build("fake_file", 100).unwrap();
    }
}

