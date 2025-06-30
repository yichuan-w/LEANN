/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use std::mem;
use std::vec::Vec;

use hashbrown::HashSet;

use crate::{
    common::{ANNResult, AlignedBoxWithSlice},
    model::{Neighbor, NeighborPriorityQueue},
    model::data_store::DiskScratchDataset,
};

use super::{PQScratch, Scratch, MAX_GRAPH_DEGREE, QUERY_ALIGNMENT_OF_T_SIZE};

// Scratch space for disk index based search.
pub struct SSDQueryScratch<T: Default + Copy, const N: usize> 
{
    // Disk scratch dataset storing fp vectors with aligned dim (N)
    pub scratch_dataset: DiskScratchDataset<T, N>,

    // The query scratch.
    pub query: AlignedBoxWithSlice<T>,

    /// The PQ Scratch.
    pub pq_scratch: Option<Box<PQScratch>>,

    // The visited set.
    pub id_scratch: HashSet<u32>,

    /// Best candidates, whose size is candidate_queue_size
    pub best_candidates: NeighborPriorityQueue,

    // Full return set.
    pub full_return_set: Vec<Neighbor>,
}

//
impl<T: Copy + Default, const N: usize> SSDQueryScratch<T, N> 
{
    pub fn new(
        visited_reserve: usize,
        candidate_queue_size: usize,
        init_pq_scratch: bool,
    ) -> ANNResult<Self> {
        let scratch_dataset = DiskScratchDataset::<T, N>::new()?;

        let query = AlignedBoxWithSlice::<T>::new(N, mem::size_of::<T>() * QUERY_ALIGNMENT_OF_T_SIZE)?;

        let id_scratch = HashSet::<u32>::with_capacity(visited_reserve);
        let full_return_set = Vec::<Neighbor>::with_capacity(visited_reserve);
        let best_candidates = NeighborPriorityQueue::with_capacity(candidate_queue_size);

        let pq_scratch = if init_pq_scratch {
            Some(Box::new(PQScratch::new(MAX_GRAPH_DEGREE, N)?))
        } else {
            None
        };

        Ok(Self {
            scratch_dataset,
            query,
            pq_scratch,
            id_scratch,
            best_candidates,
            full_return_set,
        })
    }

    pub fn pq_scratch(&mut self) -> &Option<Box<PQScratch>> {
        &self.pq_scratch
    }
}

impl<T: Default + Copy, const N: usize> Scratch for SSDQueryScratch<T, N> 
{
    fn clear(&mut self) {
        self.id_scratch.clear();
        self.best_candidates.clear();
        self.full_return_set.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Arrange
        let visited_reserve = 100;
        let candidate_queue_size = 10;
        let init_pq_scratch = true;

        // Act
        let result =
            SSDQueryScratch::<u32, 3>::new(visited_reserve, candidate_queue_size, init_pq_scratch);

        // Assert
        assert!(result.is_ok());

        let scratch = result.unwrap();

        // Assert the properties of the scratch instance
        assert!(scratch.pq_scratch.is_some());
        assert!(scratch.id_scratch.is_empty());
        assert!(scratch.best_candidates.size() == 0);
        assert!(scratch.full_return_set.is_empty());
    }

    #[test]
    fn test_clear() {
        // Arrange
        let mut scratch = SSDQueryScratch::<u32, 3>::new(100, 10, true).unwrap();

        // Add some data to scratch fields
        scratch.id_scratch.insert(1);
        scratch.best_candidates.insert(Neighbor::new(2, 0.5));
        scratch.full_return_set.push(Neighbor::new(3, 0.8));

        // Act
        scratch.clear();

        // Assert
        assert!(scratch.id_scratch.is_empty());
        assert!(scratch.best_candidates.size() == 0);
        assert!(scratch.full_return_set.is_empty());
    }
}
