/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::mem::size_of;

use crate::common::{ANNResult, AlignedBoxWithSlice};

const MAX_PQ_CHUNKS: usize = 512;

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned pq table dist scratch, must be at least [256 * NCHUNKS]
    pub aligned_pqtable_dist_scratch: AlignedBoxWithSlice<f32>,
    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    pub aligned_dist_scratch: AlignedBoxWithSlice<f32>,
    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    pub aligned_pq_coord_scratch: AlignedBoxWithSlice<u8>,
    /// Rotated query
    pub rotated_query: AlignedBoxWithSlice<f32>,
    /// Aligned query float
    pub aligned_query_float: AlignedBoxWithSlice<f32>,
}

impl PQScratch {
    const ALIGNED_ALLOC_256: usize = 256;

    /// Create a new pq scratch
    pub fn new(graph_degree: usize, aligned_dim: usize) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            AlignedBoxWithSlice::new(graph_degree * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_pqtable_dist_scratch =
            AlignedBoxWithSlice::new(256 * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_dist_scratch =
            AlignedBoxWithSlice::new(graph_degree, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_query_float = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;
        let rotated_query = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            rotated_query,
            aligned_query_float,
        })
    }

    /// Set rotated_query and aligned_query_float values
    pub fn set<T>(&mut self, dim: usize, query: &[T], norm: f32)
    where
        T: Into<f32> + Copy,
    {
        for (d, item) in query.iter().enumerate().take(dim) {
            let query_val: f32 = (*item).into();
            if (norm - 1.0).abs() > f32::EPSILON {
                self.rotated_query[d] = query_val / norm;
                self.aligned_query_float[d] = query_val / norm;
            } else {
                self.rotated_query[d] = query_val;
                self.aligned_query_float[d] = query_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::PQScratch;

    #[test]
    fn test_pq_scratch() {
        let graph_degree = 512;
        let aligned_dim = 8;

        let mut pq_scratch: PQScratch = PQScratch::new(graph_degree, aligned_dim).unwrap();

        // Check alignment
        assert_eq!(
            (pq_scratch.aligned_pqtable_dist_scratch.as_ptr() as usize) % 256,
            0
        );
        assert_eq!((pq_scratch.aligned_dist_scratch.as_ptr() as usize) % 256, 0);
        assert_eq!(
            (pq_scratch.aligned_pq_coord_scratch.as_ptr() as usize) % 256,
            0
        );
        assert_eq!((pq_scratch.rotated_query.as_ptr() as usize) % 32, 0);
        assert_eq!((pq_scratch.aligned_query_float.as_ptr() as usize) % 32, 0);

        // Test set() method
        let query = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let norm = 2.0f32;
        pq_scratch.set::<u8>(query.len(), &query, norm);

        (0..query.len()).for_each(|i| {
            assert_eq!(pq_scratch.rotated_query[i], query[i] as f32 / norm);
            assert_eq!(pq_scratch.aligned_query_float[i], query[i] as f32 / norm);
        });
    }
}
