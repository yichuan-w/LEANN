/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Scratch space for in-memory index based search

use std::cmp::max;
use std::mem;

use hashbrown::HashSet;

use crate::common::{ANNError, ANNResult, AlignedBoxWithSlice};
use crate::model::configuration::index_write_parameters::IndexWriteParameters;
use crate::model::{Neighbor, NeighborPriorityQueue, PQScratch};

use super::Scratch;

/// In-mem index related limits
pub const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;

/// Max number of points for using bitset
pub const MAX_POINTS_FOR_USING_BITSET: usize = 100000;

/// TODO: SSD Index related limits
pub const MAX_GRAPH_DEGREE: usize = 512;

/// TODO: SSD Index related limits
pub const MAX_N_CMPS: usize = 16384;

/// TODO: SSD Index related limits
pub const SECTOR_LEN: usize = 4096;

/// TODO: SSD Index related limits
pub const MAX_N_SECTOR_READS: usize = 128;

/// The alignment required for memory access. This will be multiplied with size of T to get the actual alignment
pub const QUERY_ALIGNMENT_OF_T_SIZE: usize = 16;

/// Scratch space for in-memory index based search
#[derive(Debug)]
pub struct InMemQueryScratch<T, const N: usize> {
    /// Size of the candidate queue
    pub candidate_size: u32,

    /// Max degree for each vertex
    pub max_degree: u32,

    /// Max occlusion size
    pub max_occlusion_size: u32,

    /// Query node
    pub query: AlignedBoxWithSlice<T>,

    /// Best candidates, whose size is candidate_queue_size
    pub best_candidates: NeighborPriorityQueue,

    /// Occlude factor
    pub occlude_factor: Vec<f32>,

    /// Visited neighbor id
    pub id_scratch: Vec<u32>,

    /// The distance between visited neighbor and query node
    pub dist_scratch: Vec<f32>,

    /// The PQ Scratch, keey it private since this class use the Box to own the memory. Use the function pq_scratch to get its reference
    pub pq_scratch: Option<Box<PQScratch>>,

    /// Buffers used in process delete, capacity increases as needed
    pub expanded_nodes_set: HashSet<u32>,

    /// Expanded neighbors
    pub expanded_neighbors_vector: Vec<Neighbor>,

    /// Occlude list
    pub occlude_list_output: Vec<u32>,

    /// RobinSet for larger dataset
    pub node_visited_robinset: HashSet<u32>,
}

impl<T: Default + Copy, const N: usize> InMemQueryScratch<T, N> {
    /// Create InMemQueryScratch instance
    pub fn new(
        search_candidate_size: u32,
        index_write_parameter: &IndexWriteParameters,
        init_pq_scratch: bool,
    ) -> ANNResult<Self> {
        let indexing_candidate_size = index_write_parameter.search_list_size;
        let max_degree = index_write_parameter.max_degree;
        let max_occlusion_size = index_write_parameter.max_occlusion_size;

        if search_candidate_size == 0 || indexing_candidate_size == 0 || max_degree == 0 || N == 0 {
            return Err(ANNError::log_index_error(format!(
                "In InMemQueryScratch, one of search_candidate_size = {}, indexing_candidate_size = {}, dim = {} or max_degree = {} is zero.", 
                search_candidate_size, indexing_candidate_size, N, max_degree)));
        }

        let query = AlignedBoxWithSlice::new(N, mem::size_of::<T>() * QUERY_ALIGNMENT_OF_T_SIZE)?;
        let pq_scratch = if init_pq_scratch {
            Some(Box::new(PQScratch::new(MAX_GRAPH_DEGREE, N)?))
        } else {
            None
        };

        let occlude_factor = Vec::with_capacity(max_occlusion_size as usize);

        let capacity = (1.5 * GRAPH_SLACK_FACTOR * (max_degree as f64)).ceil() as usize;
        let id_scratch = Vec::with_capacity(capacity);
        let dist_scratch = Vec::with_capacity(capacity);

        let expanded_nodes_set = HashSet::<u32>::new();
        let expanded_neighbors_vector = Vec::<Neighbor>::new();
        let occlude_list_output = Vec::<u32>::new();

        let candidate_size = max(search_candidate_size, indexing_candidate_size);
        let node_visited_robinset = HashSet::<u32>::with_capacity(20 * candidate_size as usize);
        let scratch = Self {
            candidate_size,
            max_degree,
            max_occlusion_size,
            query,
            best_candidates: NeighborPriorityQueue::with_capacity(candidate_size as usize),
            occlude_factor,
            id_scratch,
            dist_scratch,
            pq_scratch,
            expanded_nodes_set,
            expanded_neighbors_vector,
            occlude_list_output,
            node_visited_robinset,
        };

        Ok(scratch)
    }

    /// Resize the scratch with new candidate size
    pub fn resize_for_new_candidate_size(&mut self, new_candidate_size: u32) {
        if new_candidate_size > self.candidate_size {
            let delta = new_candidate_size - self.candidate_size;
            self.candidate_size = new_candidate_size;
            self.best_candidates.reserve(delta as usize);
            self.node_visited_robinset.reserve((20 * delta) as usize);
        }
    }
}

impl<T: Default + Copy, const N: usize> Scratch for InMemQueryScratch<T, N> {
    fn clear(&mut self) {
        self.best_candidates.clear();
        self.occlude_factor.clear();

        self.node_visited_robinset.clear();

        self.id_scratch.clear();
        self.dist_scratch.clear();

        self.expanded_nodes_set.clear();
        self.expanded_neighbors_vector.clear();
        self.occlude_list_output.clear();
    }
}

#[cfg(test)]
mod inmemory_query_scratch_test {
    use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;

    use super::*;

    #[test]
    fn node_visited_robinset_test() {
        let index_write_parameter = IndexWriteParametersBuilder::new(10, 10)
            .with_max_occlusion_size(5)
            .build();

        let mut scratch =
            InMemQueryScratch::<f32, 32>::new(100, &index_write_parameter, false).unwrap();

        assert_eq!(scratch.node_visited_robinset.len(), 0);

        scratch.clear();
        assert_eq!(scratch.node_visited_robinset.len(), 0);
    }
}
