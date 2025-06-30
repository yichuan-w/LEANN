/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use logger::logger::indexlog::DiskIndexConstructionCheckpoint;
use logger::logger::indexlog::DiskIndexConstructionLog;
use logger::logger::indexlog::Log;
use logger::logger::indexlog::LogLevel;
use logger::message_handler::send_log;

use crate::{utils::Timer, common::ANNResult};

pub struct DiskIndexBuildLogger {
    timer: Timer,
    checkpoint: DiskIndexConstructionCheckpoint,
}

impl DiskIndexBuildLogger {
    pub fn new(checkpoint: DiskIndexConstructionCheckpoint) -> Self {
        Self { 
            timer: Timer::new(),
            checkpoint,
        }
    }

    pub fn log_checkpoint(&mut self, next_checkpoint: DiskIndexConstructionCheckpoint) -> ANNResult<()> {
        if self.checkpoint == DiskIndexConstructionCheckpoint::None {
            return Ok(());
        }

        let mut log = Log::default();
        let disk_index_construction_log = DiskIndexConstructionLog {
            checkpoint: self.checkpoint as i32,
            time_spent_in_seconds: self.timer.elapsed().as_secs_f32(),
            g_cycles_spent: self.timer.elapsed_gcycles(),
            log_level: LogLevel::Info as i32,
        };
        log.disk_index_construction_log = Some(disk_index_construction_log);

        send_log(log)?;
        self.checkpoint = next_checkpoint;
        self.timer.reset();
        Ok(())
    }
}

#[cfg(test)]
mod dataset_test {
    use super::*;

    #[test]
    fn test_log() {
        let mut logger = DiskIndexBuildLogger::new(DiskIndexConstructionCheckpoint::PqConstruction);
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::InmemIndexBuild).unwrap();logger.log_checkpoint(logger::logger::indexlog::DiskIndexConstructionCheckpoint::DiskLayout).unwrap();
    }
}

