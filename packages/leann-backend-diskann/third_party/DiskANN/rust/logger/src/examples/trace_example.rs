/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use log::{debug, info, log_enabled, warn, Level};
use logger::trace_logger::TraceLogger;

// cargo run --example trace_example

fn main() {
    static LOGGER: TraceLogger = TraceLogger {};
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(log::LevelFilter::Trace))
        .unwrap();

    info!("Rust logging n = {}", 42);
    warn!("This is too much fun!");
    debug!("Maybe we can make this code work");

    let error_is_enabled = log_enabled!(Level::Error);
    let warn_is_enabled = log_enabled!(Level::Warn);
    let info_is_enabled = log_enabled!(Level::Info);
    let debug_is_enabled = log_enabled!(Level::Debug);
    let trace_is_enabled = log_enabled!(Level::Trace);
    println!(
        "is_enabled?  error: {:5?}, warn: {:5?}, info: {:5?}, debug: {:5?}, trace: {:5?}",
        error_is_enabled, warn_is_enabled, info_is_enabled, debug_is_enabled, trace_is_enabled,
    );
}

