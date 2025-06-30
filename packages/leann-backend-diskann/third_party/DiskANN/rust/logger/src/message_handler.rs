/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::log_error::LogError;
use crate::logger::indexlog::DiskIndexConstructionCheckpoint;
use crate::logger::indexlog::Log;
use crate::logger::indexlog::LogLevel;

use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;
use std::thread;

use win_etw_macros::trace_logging_provider;

trait MessagePublisher {
    fn publish(&self, log_level: LogLevel, message: &str);
}

// ETW provider - the GUID specified here is that of the default provider for Geneva Metric Extensions
// We are just using it as a placeholder until we have a version of OpenTelemetry exporter for Rust
#[trace_logging_provider(guid = "edc24920-e004-40f6-a8e1-0e6e48f39d84")]
trait EtwTraceProvider {
    fn write(msg: &str);
}

struct EtwPublisher {
    provider: EtwTraceProvider,
    publish_to_stdout: bool,
}

impl EtwPublisher {
    pub fn new() -> Result<Self, win_etw_provider::Error> {
        let provider = EtwTraceProvider::new();
        Ok(EtwPublisher {
            provider,
            publish_to_stdout: true,
        })
    }
}

fn log_level_to_etw(level: LogLevel) -> win_etw_provider::Level {
    match level {
        LogLevel::Error => win_etw_provider::Level::ERROR,
        LogLevel::Warn => win_etw_provider::Level::WARN,
        LogLevel::Info => win_etw_provider::Level::INFO,
        LogLevel::Debug => win_etw_provider::Level::VERBOSE,
        LogLevel::Trace => win_etw_provider::Level(6),
        LogLevel::Unspecified => win_etw_provider::Level(6),
    }
}

fn i32_to_log_level(value: i32) -> LogLevel {
    match value {
        0 => LogLevel::Unspecified,
        1 => LogLevel::Error,
        2 => LogLevel::Warn,
        3 => LogLevel::Info,
        4 => LogLevel::Debug,
        5 => LogLevel::Trace,
        _ => LogLevel::Unspecified,
    }
}

impl MessagePublisher for EtwPublisher {
    fn publish(&self, log_level: LogLevel, message: &str) {
        let options = win_etw_provider::EventOptions {
            level: Some(log_level_to_etw(log_level)),
            ..Default::default()
        };
        self.provider.write(Some(&options), message);

        if self.publish_to_stdout {
            println!("{}", message);
        }
    }
}

struct MessageProcessor {
    sender: Mutex<Sender<Log>>,
}

impl MessageProcessor {
    pub fn start_processing() -> Self {
        let (sender, receiver) = mpsc::channel::<Log>();
        thread::spawn(move || -> Result<(), LogError> {
            for message in receiver {
                // Process the received message
                if let Some(indexlog) = message.index_construction_log {
                    let str = format!(
                        "Time for {}% of index build completed: {:.3} seconds, {:.3}B cycles",
                        indexlog.percentage_complete,
                        indexlog.time_spent_in_seconds,
                        indexlog.g_cycles_spent
                    );
                    publish(i32_to_log_level(indexlog.log_level), &str)?;
                }

                if let Some(disk_index_log) = message.disk_index_construction_log {
                    let str = format!(
                        "Time for disk index build [Checkpoint: {:?}] completed: {:.3} seconds, {:.3}B cycles",
                        DiskIndexConstructionCheckpoint::from_i32(disk_index_log.checkpoint).unwrap_or(DiskIndexConstructionCheckpoint::None),
                        disk_index_log.time_spent_in_seconds,
                        disk_index_log.g_cycles_spent
                    );
                    publish(i32_to_log_level(disk_index_log.log_level), &str)?;
                }

                if let Some(tracelog) = message.trace_log {
                    let str = format!("{}:{}", tracelog.log_level, tracelog.log_line);
                    publish(i32_to_log_level(tracelog.log_level), &str)?;
                }

                if let Some(err) = message.error_log {
                    publish(i32_to_log_level(err.log_level), &err.error_message)?;
                }
            }

            Ok(())
        });

        let sender = Mutex::new(sender);
        MessageProcessor { sender }
    }

    /// Log the message.
    fn log(&self, message: Log) -> Result<(), LogError> {
        Ok(self
            .sender
            .lock()
            .map_err(|err| LogError::LockPoisonError {
                err: err.to_string(),
            })?
            .send(message)?)
    }
}

lazy_static::lazy_static! {
    /// Singleton logger.
    static ref PROCESSOR: MessageProcessor = {

        MessageProcessor::start_processing()
    };
}

lazy_static::lazy_static! {
    /// Singleton publisher.
    static ref PUBLISHER: Result<EtwPublisher, win_etw_provider::Error> = {
        EtwPublisher::new()
    };
}

/// Send a message to the logging system.
pub fn send_log(message: Log) -> Result<(), LogError> {
    PROCESSOR.log(message)
}

fn publish(log_level: LogLevel, message: &str) -> Result<(), LogError> {
    match *PUBLISHER {
        Ok(ref etw_publisher) => {
            etw_publisher.publish(log_level, message);
            Ok(())
        }
        Err(ref err) => Err(LogError::ETWProviderError { err: err.clone() }),
    }
}

