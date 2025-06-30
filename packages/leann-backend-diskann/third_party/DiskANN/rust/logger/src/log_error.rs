/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::mpsc::SendError;

use crate::logger::indexlog::Log;

#[derive(thiserror::Error, Debug, Clone)]
pub enum LogError {
    /// Sender failed to send message to the channel
    #[error("IOError: {err}")]
    SendError {
        #[from]
        err: SendError<Log>,
    },

    /// PoisonError which can be returned whenever a lock is acquired
    /// Both Mutexes and RwLocks are poisoned whenever a thread fails while the lock is held
    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    /// Failed to create EtwPublisher
    #[error("EtwProviderError: {err:?}")]
    ETWProviderError { err: win_etw_provider::Error },
}

