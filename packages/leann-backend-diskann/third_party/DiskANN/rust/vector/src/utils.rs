/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Prefetch the given vector in chunks of 64 bytes, which is a cache line size
/// NOTE: good efficiency when total_vec_size is integral multiple of 64
#[inline]
pub fn prefetch_vector<T>(vec: &[T]) {
    let vec_ptr = vec.as_ptr() as *const i8;
    let vecsize = std::mem::size_of_val(vec);
    let max_prefetch_size = (vecsize / 64) * 64;

    for d in (0..max_prefetch_size).step_by(64) {
        unsafe {
            _mm_prefetch(vec_ptr.add(d), _MM_HINT_T0);
        }
    }
}

