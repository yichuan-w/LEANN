/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::atomic::{AtomicUsize, Ordering};

use logger::logger::indexlog::IndexConstructionLog;
use logger::logger::indexlog::Log;
use logger::logger::indexlog::LogLevel;
use logger::message_handler::send_log;

use crate::common::ANNResult;
use crate::utils::Timer;

pub struct IndexLogger {
    items_processed: AtomicUsize,
    timer: Timer,
    range: usize,
}

impl IndexLogger {
    pub fn new(range: usize) -> Self {
        Self {
            items_processed: AtomicUsize::new(0),
            timer: Timer::new(),
            range,
        }
    }

    pub fn vertex_processed(&self) -> ANNResult<()> {
        let count = self.items_processed.fetch_add(1, Ordering::Relaxed);
        if count % 100_000 == 0 {
            let mut log = Log::default();
            let index_construction_log = IndexConstructionLog {
                    percentage_complete: (100_f32 * count as f32) / (self.range as f32),
                    time_spent_in_seconds: self.timer.elapsed().as_secs_f32(),
                    g_cycles_spent: self.timer.elapsed_gcycles(),
                    log_level: LogLevel::Info as i32,
                };
            log.index_construction_log = Some(index_construction_log);

            send_log(log)?;
        }

        Ok(())
    }
}
