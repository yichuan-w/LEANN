/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub mod perf;
pub use perf::{get_process_cycle_time, get_process_handle};

pub mod file_io;
pub use file_io::{get_queued_completion_status, read_file_to_slice};

pub mod file_handle;
pub use file_handle::FileHandle;

pub mod io_completion_port;
pub use io_completion_port::IOCompletionPort;
