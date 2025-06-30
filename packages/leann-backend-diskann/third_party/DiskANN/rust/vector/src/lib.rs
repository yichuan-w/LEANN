/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

// #![feature(stdsimd)]
// mod f32x16;
// Uncomment above 2 to experiment with f32x16
mod distance;
mod half;
mod l2_float_distance;
mod metric;
mod utils;

pub use crate::half::Half;
pub use distance::FullPrecisionDistance;
pub use metric::Metric;
pub use utils::prefetch_vector;

#[cfg(test)]
mod distance_test;
mod test_util;
