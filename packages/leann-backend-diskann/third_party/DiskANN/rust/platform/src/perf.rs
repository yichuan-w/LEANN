/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: bool, dwProcessId: u32) -> usize;
    fn QueryProcessCycleTime(hProcess: usize, lpCycleTime: *mut u64) -> bool;
    fn GetCurrentProcessId() -> u32;
}

/// Get current process handle.
pub fn get_process_handle() -> Option<usize> {
    if cfg!(windows) {
        const PROCESS_QUERY_INFORMATION: u32 = 0x0400;
        const PROCESS_VM_READ: u32 = 0x0010;

        unsafe {
            let current_process_id = GetCurrentProcessId();
            let handle = OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                false,
                current_process_id,
            );
            if handle == 0 {
                None
            } else {
                Some(handle)
            }
        }
    } else {
        None
    }
}

pub fn get_process_cycle_time(process_handle: Option<usize>) -> Option<u64> {
    let mut cycle_time: u64 = 0;
    if cfg!(windows) {
        if let Some(handle) = process_handle {
            let result = unsafe { QueryProcessCycleTime(handle, &mut cycle_time as *mut u64) };
            if result {
                return Some(cycle_time);
            }
        }
    }

    None
}

