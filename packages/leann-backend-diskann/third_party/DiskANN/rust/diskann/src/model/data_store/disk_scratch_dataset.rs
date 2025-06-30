/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Disk scratch dataset

use std::mem::{size_of, size_of_val};
use std::ptr;

use crate::common::{AlignedBoxWithSlice, ANNResult};
use crate::model::MAX_N_CMPS;
use crate::utils::round_up;

/// DiskScratchDataset alignment
pub const DISK_SCRATCH_DATASET_ALIGN: usize = 256;

/// Disk scratch dataset storing fp vectors with aligned dim
#[derive(Debug)]
pub struct DiskScratchDataset<T, const N: usize>
{
    /// fp vectors with aligned dim
    pub data: AlignedBoxWithSlice<T>, 

    /// current index to store the next fp vector
    pub cur_index: usize,
}

impl<T, const N: usize> DiskScratchDataset<T, N>
{
    /// Create DiskScratchDataset instance
    pub fn new() -> ANNResult<Self> {
        Ok(Self {
            // C++ code allocates round_up(MAX_N_CMPS * N, 256) bytes, shouldn't it be round_up(MAX_N_CMPS * N, 256) * size_of::<T> bytes?
            data: AlignedBoxWithSlice::new(
                round_up(MAX_N_CMPS * N, DISK_SCRATCH_DATASET_ALIGN), 
                DISK_SCRATCH_DATASET_ALIGN)?,
            cur_index: 0,
        })
    }

    /// memcpy from fp vector bytes (its len should be `dim * size_of::<T>()`) to self.data
    /// The dest slice is a fp vector with aligned dim
    /// * fp_vector_buf's dim might not be aligned dim (N)
    /// # Safety
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * `fp_vector_buf`'s len must be `dim * size_of::<T>()` bytes
    /// 
    /// * `fp_vector_buf` must be smaller than or equal to `N * size_of::<T>()` bytes.
    ///
    /// * `fp_vector_buf` and `self.data` must be nonoverlapping.
    pub unsafe fn memcpy_from_fp_vector_buf(&mut self, fp_vector_buf: &[u8]) -> &[T] {
        if self.cur_index == MAX_N_CMPS {
            self.cur_index = 0;
        }

        let aligned_dim_vector = &mut self.data[self.cur_index * N..(self.cur_index + 1) * N];

        assert!(fp_vector_buf.len() % size_of::<T>() == 0);
        assert!(fp_vector_buf.len() <= size_of_val(aligned_dim_vector));

        // memcpy from fp_vector_buf to aligned_dim_vector
        unsafe {
            ptr::copy_nonoverlapping(
                fp_vector_buf.as_ptr(),
                aligned_dim_vector.as_mut_ptr() as *mut u8,
                fp_vector_buf.len(),
            );
        }

        self.cur_index += 1;
        aligned_dim_vector
    }
}
