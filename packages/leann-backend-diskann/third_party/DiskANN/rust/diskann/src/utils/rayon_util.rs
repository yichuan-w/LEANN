/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::ops::Range;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::common::ANNResult;

/// based on thread_num, execute the task in parallel using Rayon or serial
#[inline]
pub fn execute_with_rayon<F>(range: Range<usize>, num_threads: u32, f: F) -> ANNResult<()>
where F: Fn(usize) -> ANNResult<()> + Sync + Send + Copy
{
    if num_threads == 1 {
        for i in range {
            f(i)?;
        }
        Ok(())
    } else {
        range.into_par_iter().try_for_each(f)
    }
}

/// set the thread count of Rayon, otherwise it will use threads as many as logical cores.
#[inline]
pub fn set_rayon_num_threads(num_threads: u32) {
    std::env::set_var(
        "RAYON_NUM_THREADS",
        num_threads.to_string(),
    );
}

