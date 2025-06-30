/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::log_error::LogError;
use crate::logger::indexlog::{ErrorLog, Log, LogLevel};
use crate::message_handler::send_log;

pub fn log_error(error_message: String) -> Result<(), LogError> {
    let mut log = Log::default();
    let error_log = ErrorLog {
        log_level: LogLevel::Error as i32,
        error_message,
    };
    log.error_log = Some(error_log);

    send_log(log)
}

#[cfg(test)]
mod error_logger_test {
    use super::*;

    #[test]
    fn log_error_works() {
        log_error(String::from("Error")).unwrap();
    }
}

