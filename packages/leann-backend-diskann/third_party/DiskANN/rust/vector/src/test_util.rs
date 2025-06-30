/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[cfg(test)]
use crate::Half;

#[cfg(test)]
pub fn no_vector_compare_f16(a: &[Half], b: &[Half]) -> f32 {
    let mut sum = 0.0;
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        sum += (a[i].to_f32() - b[i].to_f32()).powi(2);
    }
    sum
}

#[cfg(test)]
pub fn no_vector_compare_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    debug_assert_eq!(a.len(), b.len());

    for i in 0..a.len() {
        sum += (a[i] - b[i]).powi(2);
    }
    sum
}

