/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]
#![cfg_attr(test, allow(clippy::unused_io_amount))]

pub mod utils;

pub mod algorithm;

pub mod model;

pub mod common;

pub mod index;

pub mod storage;

pub mod instrumentation;

#[cfg(test)]
pub mod test_utils;
