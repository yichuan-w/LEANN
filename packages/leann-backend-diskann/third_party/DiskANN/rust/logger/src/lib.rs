/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub mod logger {
    pub mod indexlog {
        include!(concat!(env!("OUT_DIR"), "/diskann_logger.rs"));
    }
}

pub mod error_logger;
pub mod log_error;
pub mod message_handler;
pub mod trace_logger;
