/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::common::ANNResult;

use super::ArcConcurrentBoxedQueue;
use super::{scratch_traits::Scratch};
use std::time::Duration;

pub struct ScratchStoreManager<T: Scratch> {
    scratch: Option<Box<T>>,
    scratch_pool: ArcConcurrentBoxedQueue<T>,
}

impl<T: Scratch> ScratchStoreManager<T> {
    pub fn new(scratch_pool: ArcConcurrentBoxedQueue<T>, wait_time: Duration) -> ANNResult<Self> {
        let mut scratch = scratch_pool.pop()?;
        while scratch.is_none() {
            scratch_pool.wait_for_push_notify(wait_time)?;
            scratch = scratch_pool.pop()?;
        }

        Ok(ScratchStoreManager {
            scratch,
            scratch_pool,
        })
    }

    pub fn scratch_space(&mut self) -> Option<&mut T> {
        self.scratch.as_deref_mut()
    }
}

impl<T: Scratch> Drop for ScratchStoreManager<T> {
    fn drop(&mut self) {
        if let Some(mut scratch) = self.scratch.take() {
            scratch.clear();
            let _ = self.scratch_pool.push(scratch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MyScratch {
        data: Vec<i32>,
    }

    impl Scratch for MyScratch {
        fn clear(&mut self) {
            self.data.clear();
        }
    }

    #[test]
    fn test_scratch_store_manager() {
        let wait_time = Duration::from_millis(100);

        let scratch_pool = ArcConcurrentBoxedQueue::new();
        for i in 1..3 {
            scratch_pool.push(Box::new(MyScratch {
                data: vec![i, 2 * i, 3 * i],
            })).unwrap();
        }

        let mut manager = ScratchStoreManager::new(scratch_pool.clone(), wait_time).unwrap();
        let scratch_space = manager.scratch_space().unwrap();

        assert_eq!(scratch_space.data, vec![1, 2, 3]);

        // At this point, the ScratchStoreManager will go out of scope,
        // causing the Drop implementation to be called, which should
        // call the clear method on MyScratch.
        drop(manager);

        let current_scratch = scratch_pool.pop().unwrap().unwrap();
        assert_eq!(current_scratch.data, vec![2, 4, 6]);
    }
}

