/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use std::sync::Arc;

use super::{scratch_traits::Scratch, IOContext, SSDQueryScratch};
use crate::common::ANNResult;

// The thread data struct for SSD I/O. One for each thread, contains the ScratchSpace and the IOContext.
pub struct SSDThreadData<T: Default + Copy, const N: usize> {
    pub scratch: SSDQueryScratch<T, N>,
    pub io_context: Option<Arc<IOContext>>,
}

impl<T: Default + Copy, const N: usize> SSDThreadData<T, N> {
    pub fn new(
        aligned_dim: usize,
        visited_reserve: usize,
        init_pq_scratch: bool,
    ) -> ANNResult<Self> {
        let scratch = SSDQueryScratch::new(aligned_dim, visited_reserve, init_pq_scratch)?;
        Ok(SSDThreadData {
            scratch,
            io_context: None,
        })
    }

    pub fn clear(&mut self) {
        self.scratch.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::model::Neighbor;

    use super::*;

    #[test]
    fn test_new() {
        // Arrange
        let aligned_dim = 10;
        let visited_reserve = 100;
        let init_pq_scratch = true;

        // Act
        let result = SSDThreadData::<u32, 3>::new(aligned_dim, visited_reserve, init_pq_scratch);

        // Assert
        assert!(result.is_ok());

        let thread_data = result.unwrap();

        // Assert the properties of the thread data instance
        assert!(thread_data.io_context.is_none());

        let scratch = &thread_data.scratch;
        // Assert the properties of the scratch instance
        assert!(scratch.pq_scratch.is_some());
        assert!(scratch.id_scratch.is_empty());
        assert!(scratch.best_candidates.size() == 0);
        assert!(scratch.full_return_set.is_empty());
    }

    #[test]
    fn test_clear() {
        // Arrange
        let mut thread_data = SSDThreadData::<u32, 3>::new(10, 100, true).unwrap();

        // Add some data to scratch fields
        thread_data.scratch.id_scratch.insert(1);
        thread_data
            .scratch
            .best_candidates
            .insert(Neighbor::new(2, 0.5));
        thread_data
            .scratch
            .full_return_set
            .push(Neighbor::new(3, 0.8));

        // Act
        thread_data.clear();

        // Assert
        assert!(thread_data.scratch.id_scratch.is_empty());
        assert!(thread_data.scratch.best_candidates.size() == 0);
        assert!(thread_data.scratch.full_return_set.is_empty());
    }
}

