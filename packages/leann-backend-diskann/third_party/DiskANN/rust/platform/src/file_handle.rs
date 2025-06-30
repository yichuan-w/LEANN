/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::ffi::CString;
use std::{io, ptr};

use winapi::um::fileapi::OPEN_EXISTING;
use winapi::um::winbase::{FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_FLAG_RANDOM_ACCESS};
use winapi::um::winnt::{FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, GENERIC_READ, GENERIC_WRITE};

use winapi::{
    shared::minwindef::DWORD,
    um::{
        errhandlingapi::GetLastError,
        fileapi::CreateFileA,
        handleapi::{CloseHandle, INVALID_HANDLE_VALUE},
        winnt::HANDLE,
    },
};

pub const FILE_ATTRIBUTE_READONLY: DWORD = 0x00000001;

/// `AccessMode` determines how a file can be accessed.
/// These modes are used when creating or opening a file to decide what operations are allowed
/// to be performed on the file.
///
/// # Variants
///
/// - `Read`: The file is opened in read-only mode.
///
/// - `Write`: The file is opened in write-only mode.
///
/// - `ReadWrite`: The file is opened for both reading and writing.
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// `ShareMode` determines how a file can be shared.
///
/// These modes are used when creating or opening a file to decide what operations other 
/// opening instances of the file can perform on it.
/// # Variants
/// - `None`: Prevents other processes from opening a file if they request delete, 
///   read, or write access.
///
/// - `Read`: Allows subsequent open operations on the same file to request read access.
///
/// - `Write`: Allows subsequent open operations on the same file file to request write access.
///
/// - `Delete`: Allows subsequent open operations on the same file file to request delete access.
pub enum ShareMode {
    None,
    Read,
    Write,
    Delete,
}

/// # Windows File Handle Wrapper
///
/// Introduces a Rust-friendly wrapper around the native Windows `HANDLE` object, `FileHandle`.
/// `FileHandle` provides safe creation and automatic cleanup of Windows file handles, leveraging Rust's ownership model.

/// `FileHandle` struct that wraps a native Windows `HANDLE` object
#[cfg(target_os = "windows")]
pub struct FileHandle {
    handle: HANDLE,
}

impl FileHandle {
    /// Creates a new `FileHandle` by opening an existing file with the given access and shared mode.
    ///
    /// This function is marked unsafe because it creates a raw pointer to the filename and try to create
    /// a Windows `HANDLE` object without checking if you have sufficient permissions.
    ///
    /// # Safety
    ///
    /// Ensure that the file specified by `file_name` is valid and the calling process has
    /// sufficient permissions to perform the specified `access_mode` and `share_mode` operations.
    ///
    /// # Parameters
    ///
    /// - `file_name`: The name of the file.
    /// - `access_mode`: The access mode to be used for the file.
    /// - `share_mode`: The share mode to be used for the file
    ///
    /// # Errors
    /// This function will return an error if the `file_name` is invalid or if the file cannot
    /// be opened with the specified `access_mode` and `share_mode`.
    pub unsafe fn new(
        file_name: &str,
        access_mode: AccessMode,
        share_mode: ShareMode,
    ) -> io::Result<Self> {
        let file_name_c = CString::new(file_name).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid file name. {}", file_name),
            )
        })?;

        let dw_desired_access = match access_mode {
            AccessMode::Read => GENERIC_READ,
            AccessMode::Write => GENERIC_WRITE,
            AccessMode::ReadWrite => GENERIC_READ | GENERIC_WRITE,
        };

        let dw_share_mode = match share_mode {
            ShareMode::None => 0,
            ShareMode::Read => FILE_SHARE_READ,
            ShareMode::Write => FILE_SHARE_WRITE,
            ShareMode::Delete => FILE_SHARE_DELETE,
        };

        let dw_flags_and_attributes = FILE_ATTRIBUTE_READONLY
            | FILE_FLAG_NO_BUFFERING
            | FILE_FLAG_OVERLAPPED
            | FILE_FLAG_RANDOM_ACCESS;

        let handle = unsafe {
            CreateFileA(
                file_name_c.as_ptr(),
                dw_desired_access,
                dw_share_mode,
                ptr::null_mut(),
                OPEN_EXISTING,
                dw_flags_and_attributes,
                ptr::null_mut(),
            )
        };

        if handle == INVALID_HANDLE_VALUE {
            let error_code = unsafe { GetLastError() };
            Err(io::Error::from_raw_os_error(error_code as i32))
        } else {
            Ok(Self { handle })
        }
    }

    pub fn raw_handle(&self) -> HANDLE {
        self.handle
    }
}

impl Drop for FileHandle {
    /// Automatically closes the `FileHandle` when it goes out of scope.
    /// Any errors in closing the handle are logged, as `Drop` does not support returning `Result`.
    fn drop(&mut self) {
        let result = unsafe { CloseHandle(self.handle) };
        if result == 0 {
            let error_code = unsafe { GetLastError() };
            let error = io::Error::from_raw_os_error(error_code as i32);

            // Only log the error if dropping the handle fails, since Rust's Drop trait does not support returning Result types from the drop method,
            // and panicking in the drop method is considered bad practice
            log::warn!("Error when dropping IOCompletionPort: {:?}", error);
        }
    }
}

/// Returns a `FileHandle` with an `INVALID_HANDLE_VALUE`.
impl Default for FileHandle {
    fn default() -> Self {
        Self {
            handle: INVALID_HANDLE_VALUE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::Path;

    #[test]
    fn test_create_file() {
        // Create a dummy file
        let dummy_file_path = "dummy_file.txt";
        {
            let _file = File::create(dummy_file_path).expect("Failed to create dummy file.");
        }

        let path = Path::new(dummy_file_path);
        {
            let file_handle = unsafe {
                FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read)
            };

            // Check that the file handle is valid
            assert!(file_handle.is_ok());
        }

        // Try to delete the file. If the handle was correctly dropped, this should succeed.
        match std::fs::remove_file(dummy_file_path) {
            Ok(()) => (), // File was deleted successfully, which means the handle was closed.
            Err(e) => panic!("Failed to delete file: {}", e), // Failed to delete the file, likely because the handle is still open.
        }
    }

    #[test]
    fn test_file_not_found() {
        let path = Path::new("non_existent_file.txt");
        let file_handle =
            unsafe { FileHandle::new(path.to_str().unwrap(), AccessMode::Read, ShareMode::Read) };

        // Check that opening a non-existent file returns an error
        assert!(file_handle.is_err());
    }
}
