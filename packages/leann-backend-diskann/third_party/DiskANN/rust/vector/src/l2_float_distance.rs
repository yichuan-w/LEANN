/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Distance calculation for L2 Metric

#[cfg(not(target_feature = "avx2"))]
compile_error!("Library must be compiled with -C target-feature=+avx2");

use std::arch::x86_64::*;

use crate::Half;

/// Calculate the distance by vector arithmetic
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);

    // make sure the addresses are bytes aligned
    debug_assert_eq!(a.as_ptr().align_offset(32), 0);
    debug_assert_eq!(b.as_ptr().align_offset(32), 0);

    unsafe {
        let mut sum = _mm256_setzero_ps();
        let a_ptr = a.as_ptr() as *const __m128i;
        let b_ptr = b.as_ptr() as *const __m128i;

        // Iterate over the elements in steps of 8
        for i in (0..N).step_by(8) {
            let a_vec = _mm256_cvtph_ps(_mm_load_si128(a_ptr.add(i / 8)));
            let b_vec = _mm256_cvtph_ps(_mm_load_si128(b_ptr.add(i / 8)));

            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        /* Conversion to float is a no-op on x86-64 */
        _mm_cvtss_f32(x32)
    }
}

/// Calculate the distance by vector arithmetic
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);

    // make sure the addresses are bytes aligned
    debug_assert_eq!(a.as_ptr().align_offset(32), 0);
    debug_assert_eq!(b.as_ptr().align_offset(32), 0);

    unsafe {
        let mut sum = _mm256_setzero_ps();

        // Iterate over the elements in steps of 8
        for i in (0..N).step_by(8) {
            let a_vec = _mm256_load_ps(&a[i]);
            let b_vec = _mm256_load_ps(&b[i]);
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        /* Conversion to float is a no-op on x86-64 */
        _mm_cvtss_f32(x32)
    }
}

