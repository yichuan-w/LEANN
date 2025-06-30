/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::logger::indexlog::{Log, TraceLog};
use crate::message_handler::send_log;

use log;

pub struct TraceLogger {}

fn level_to_i32(value: log::Level) -> i32 {
    match value {
        log::Level::Error => 1,
        log::Level::Warn => 2,
        log::Level::Info => 3,
        log::Level::Debug => 4,
        log::Level::Trace => 5,
    }
}

impl log::Log for TraceLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        let message = record.args().to_string();
        let metadata = record.metadata();
        let mut log = Log::default();
        let trace_log = TraceLog {
            log_line: message,
            log_level: level_to_i32(metadata.level()),
        };
        log.trace_log = Some(trace_log);
        let _ = send_log(log);
    }

    fn flush(&self) {}
}

