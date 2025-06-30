/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::alloc::LayoutError;
use std::array::TryFromSliceError;
use std::io;
use std::num::TryFromIntError;

use logger::error_logger::log_error;
use logger::log_error::LogError;

/// Result
pub type ANNResult<T> = Result<T, ANNError>;

/// DiskANN Error
/// ANNError is `Send` (i.e., safe to send across threads)
#[derive(thiserror::Error, Debug)]
pub enum ANNError {
    /// Index construction and search error
    #[error("IndexError: {err}")]
    IndexError { err: String },

    /// Index configuration error
    #[error("IndexConfigError: {parameter} is invalid, err={err}")]
    IndexConfigError { parameter: String, err: String },

    /// Integer conversion error
    #[error("TryFromIntError: {err}")]
    TryFromIntError {
        #[from]
        err: TryFromIntError,
    },

    /// IO error
    #[error("IOError: {err}")]
    IOError {
        #[from]
        err: io::Error,
    },

    /// Layout error in memory allocation
    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayoutError {
        #[from]
        err: LayoutError,
    },

    /// PoisonError which can be returned whenever a lock is acquired
    /// Both Mutexes and RwLocks are poisoned whenever a thread fails while the lock is held
    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    /// DiskIOAlignmentError which can be returned when calling windows API CreateFileA for the disk index file fails.
    #[error("DiskIOAlignmentError: {err}")]
    DiskIOAlignmentError { err: String },

    /// Logging error
    #[error("LogError: {err}")]
    LogError {
        #[from]
        err: LogError,
    },

    // PQ construction error
    // Error happened when we construct PQ pivot or PQ compressed table
    #[error("PQError: {err}")]
    PQError { err: String },

    /// Array conversion error
    #[error("Error try creating array from slice: {err}")]
    TryFromSliceError {
        #[from]
        err: TryFromSliceError,
    },
}

impl ANNError {
    /// Create, log and return IndexError
    #[inline]
    pub fn log_index_error(err: String) -> Self {
        let ann_err = ANNError::IndexError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return IndexConfigError
    #[inline]
    pub fn log_index_config_error(parameter: String, err: String) -> Self {
        let ann_err = ANNError::IndexConfigError { parameter, err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return TryFromIntError
    #[inline]
    pub fn log_try_from_int_error(err: TryFromIntError) -> Self {
        let ann_err = ANNError::TryFromIntError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return IOError
    #[inline]
    pub fn log_io_error(err: io::Error) -> Self {
        let ann_err = ANNError::IOError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return DiskIOAlignmentError
    /// #[inline]
    pub fn log_disk_io_request_alignment_error(err: String) -> Self {
        let ann_err: ANNError = ANNError::DiskIOAlignmentError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return IOError
    #[inline]
    pub fn log_mem_alloc_layout_error(err: LayoutError) -> Self {
        let ann_err = ANNError::MemoryAllocLayoutError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return LockPoisonError
    #[inline]
    pub fn log_lock_poison_error(err: String) -> Self {
        let ann_err = ANNError::LockPoisonError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return PQError
    #[inline]
    pub fn log_pq_error(err: String) -> Self {
        let ann_err = ANNError::PQError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    /// Create, log and return TryFromSliceError
    #[inline]
    pub fn log_try_from_slice_error(err: TryFromSliceError) -> Self {
        let ann_err = ANNError::TryFromSliceError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }
}

#[cfg(test)]
mod ann_result_test {
    use super::*;

    #[test]
    fn ann_err_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ANNError>();
    }
}
