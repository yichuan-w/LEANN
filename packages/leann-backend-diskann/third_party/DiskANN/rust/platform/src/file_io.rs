/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
/// The module provides unsafe wrappers around two Windows API functions: `ReadFile` and `GetQueuedCompletionStatus`.
///
/// These wrappers aim to simplify and abstract the use of these functions, providing easier error handling and a safer interface.
/// They return standard Rust `io::Result` types for convenience and consistency with the rest of the Rust standard library.
use std::io;
use std::ptr;

use winapi::{
    ctypes::c_void,
    shared::{
        basetsd::ULONG_PTR,
        minwindef::{DWORD, FALSE},
        winerror::{ERROR_IO_PENDING, WAIT_TIMEOUT},
    },
    um::{
        errhandlingapi::GetLastError, fileapi::ReadFile, ioapiset::GetQueuedCompletionStatus,
        minwinbase::OVERLAPPED,
    },
};

use crate::FileHandle;
use crate::IOCompletionPort;

/// Asynchronously queue a read request from a file into a buffer slice.
///
/// Wraps the unsafe Windows API function `ReadFile`, making it safe to call only when the overlapped buffer
/// remains valid and unchanged anywhere else during the entire async operation.
///
/// Returns a boolean indicating whether the read operation completed synchronously or is pending.
///
/// # Safety
///
/// This function is marked as `unsafe` because it uses raw pointers and requires the caller to ensure
/// that the buffer slice and the overlapped buffer stay valid during the whole async operation.
pub unsafe fn read_file_to_slice<T>(
    file_handle: &FileHandle,
    buffer_slice: &mut [T],
    overlapped: *mut OVERLAPPED,
    offset: u64,
) -> io::Result<bool> {
    let num_bytes = std::mem::size_of_val(buffer_slice);
    unsafe {
        ptr::write(overlapped, std::mem::zeroed());
        (*overlapped).u.s_mut().Offset = offset as u32;
        (*overlapped).u.s_mut().OffsetHigh = (offset >> 32) as u32;
    }

    let result = unsafe {
        ReadFile(
            file_handle.raw_handle(),
            buffer_slice.as_mut_ptr() as *mut c_void,
            num_bytes as DWORD,
            ptr::null_mut(),
            overlapped,
        )
    };

    match result {
        FALSE => {
            let error = unsafe { GetLastError() };
            if error != ERROR_IO_PENDING {
                Err(io::Error::from_raw_os_error(error as i32))
            } else {
                Ok(false)
            }
        }
        _ => Ok(true),
    }
}

/// Retrieves the results of an asynchronous I/O operation on an I/O completion port.
///
/// Wraps the unsafe Windows API function `GetQueuedCompletionStatus`, making it safe to call only when the overlapped buffer
/// remains valid and unchanged anywhere else during the entire async operation.
///
/// Returns a boolean indicating whether an I/O operation completed synchronously or is still pending.
///
/// # Safety
///
/// This function is marked as `unsafe` because it uses raw pointers and requires the caller to ensure
/// that the overlapped buffer stays valid during the whole async operation.
pub unsafe fn get_queued_completion_status(
    completion_port: &IOCompletionPort,
    lp_number_of_bytes: &mut DWORD,
    lp_completion_key: &mut ULONG_PTR,
    lp_overlapped: *mut *mut OVERLAPPED,
    dw_milliseconds: DWORD,
) -> io::Result<bool> {
    let result = unsafe {
        GetQueuedCompletionStatus(
            completion_port.raw_handle(),
            lp_number_of_bytes,
            lp_completion_key,
            lp_overlapped,
            dw_milliseconds,
        )
    };

    match result {
        0 => {
            let error = unsafe { GetLastError() };
            if error == WAIT_TIMEOUT {
                Ok(false)
            } else {
                Err(io::Error::from_raw_os_error(error as i32))
            }
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use crate::file_handle::{AccessMode, ShareMode};

    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_read_file_to_slice() {
        // Create a temporary file and write some data into it
        let path = Path::new("temp.txt");
        {
            let mut file = File::create(path).unwrap();
            file.write_all(b"Hello, world!").unwrap();
        }

        let mut buffer: [u8; 512] = [0; 512];
        let mut overlapped = unsafe { std::mem::zeroed::<OVERLAPPED>() };
        {
            let file_handle = unsafe {
                FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read)
            }
            .unwrap();

            // Call the function under test
            let result =
                unsafe { read_file_to_slice(&file_handle, &mut buffer, &mut overlapped, 0) };

            assert!(result.is_ok());
            let result_str = std::str::from_utf8(&buffer[.."Hello, world!".len()]).unwrap();
            assert_eq!(result_str, "Hello, world!");
        }

        // Clean up
        std::fs::remove_file("temp.txt").unwrap();
    }
}
