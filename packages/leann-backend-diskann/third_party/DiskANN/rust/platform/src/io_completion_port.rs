/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::io;

use winapi::{
    ctypes::c_void,
    shared::{basetsd::ULONG_PTR, minwindef::DWORD},
    um::{
        errhandlingapi::GetLastError,
        handleapi::{CloseHandle, INVALID_HANDLE_VALUE},
        ioapiset::CreateIoCompletionPort,
        winnt::HANDLE,
    },
};

use crate::FileHandle;

/// This module provides a safe and idiomatic Rust interface over the IOCompletionPort handle and associated Windows API functions.
/// This struct represents an I/O completion port, which is an object used in asynchronous I/O operations on Windows.
pub struct IOCompletionPort {
    io_completion_port: HANDLE,
}

impl IOCompletionPort {
    /// Create a new IOCompletionPort.
    /// This function wraps the Windows CreateIoCompletionPort function, providing error handling and automatic resource management.
    ///
    /// # Arguments
    ///
    /// * `file_handle` - A reference to a FileHandle to associate with the IOCompletionPort.
    /// * `existing_completion_port` - An optional reference to an existing IOCompletionPort. If provided, the new IOCompletionPort will be associated with it.
    /// * `completion_key` - The completion key associated with the file handle.
    /// * `number_of_concurrent_threads` - The maximum number of threads that the operating system can allow to concurrently process I/O completion packets for the I/O completion port.
    ///
    /// # Return
    ///
    /// Returns a Result with the new IOCompletionPort if successful, or an io::Error if the function fails.
    pub fn new(
        file_handle: &FileHandle,
        existing_completion_port: Option<&IOCompletionPort>,
        completion_key: ULONG_PTR,
        number_of_concurrent_threads: DWORD,
    ) -> io::Result<Self> {
        let io_completion_port = unsafe {
            CreateIoCompletionPort(
                file_handle.raw_handle(),
                existing_completion_port
                    .map_or(std::ptr::null_mut::<c_void>(), |io_completion_port| {
                        io_completion_port.raw_handle()
                    }),
                completion_key,
                number_of_concurrent_threads,
            )
        };

        if io_completion_port == INVALID_HANDLE_VALUE {
            let error_code = unsafe { GetLastError() };
            return Err(io::Error::from_raw_os_error(error_code as i32));
        }

        Ok(IOCompletionPort { io_completion_port })
    }

    pub fn raw_handle(&self) -> HANDLE {
        self.io_completion_port
    }
}

impl Drop for IOCompletionPort {
    /// Drop method for IOCompletionPort.
    /// This wraps the Windows CloseHandle function, providing automatic resource cleanup when the IOCompletionPort is dropped.
    /// If an error occurs while dropping, it is logged and the drop continues. This is because panicking in Drop can cause unwinding issues.
    fn drop(&mut self) {
        let result = unsafe { CloseHandle(self.io_completion_port) };
        if result == 0 {
            let error_code = unsafe { GetLastError() };
            let error = io::Error::from_raw_os_error(error_code as i32);

            // Only log the error if dropping the handle fails, since Rust's Drop trait does not support returning Result types from the drop method,
            // and panicking in the drop method is considered bad practice
            log::warn!("Error when dropping IOCompletionPort: {:?}", error);
        }
    }
}

impl Default for IOCompletionPort {
    /// Create a default IOCompletionPort, whose handle is set to INVALID_HANDLE_VALUE.
    /// Returns a new IOCompletionPort with handle set to INVALID_HANDLE_VALUE.
    fn default() -> Self {
        Self {
            io_completion_port: INVALID_HANDLE_VALUE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_handle::{AccessMode, ShareMode};

    #[test]
    fn create_io_completion_port() {
        let file_name = "../diskann/tests/data/delete_set_50pts.bin";
        let file_handle = unsafe { FileHandle::new(file_name, AccessMode::Read, ShareMode::Read) }
            .expect("Failed to create file handle.");

        let io_completion_port = IOCompletionPort::new(&file_handle, None, 0, 0);

        assert!(
            io_completion_port.is_ok(),
            "Failed to create IOCompletionPort."
        );
    }

    #[test]
    fn drop_io_completion_port() {
        let file_name = "../diskann/tests/data/delete_set_50pts.bin";
        let file_handle = unsafe { FileHandle::new(file_name, AccessMode::Read, ShareMode::Read) }
            .expect("Failed to create file handle.");

        let io_completion_port = IOCompletionPort::new(&file_handle, None, 0, 0)
            .expect("Failed to create IOCompletionPort.");

        // After this line, io_completion_port goes out of scope and its Drop trait will be called.
        let _ = io_completion_port;
        // We have no easy way to test that the Drop trait works correctly, but if it doesn't,
        // a resource leak or other problem may become apparent in later tests or in real use of the code.
    }

    #[test]
    fn default_io_completion_port() {
        let io_completion_port = IOCompletionPort::default();
        assert_eq!(
            io_completion_port.raw_handle(),
            INVALID_HANDLE_VALUE,
            "Default IOCompletionPort did not have INVALID_HANDLE_VALUE."
        );
    }
}

