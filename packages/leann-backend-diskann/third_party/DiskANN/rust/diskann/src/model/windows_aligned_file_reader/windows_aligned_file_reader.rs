/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::Arc;
use std::time::Duration;
use std::{ptr, thread};

use crossbeam::sync::ShardedLock;
use hashbrown::HashMap;
use once_cell::sync::Lazy;

use platform::file_handle::{AccessMode, ShareMode};
use platform::{
    file_handle::FileHandle,
    file_io::{get_queued_completion_status, read_file_to_slice},
    io_completion_port::IOCompletionPort,
};

use winapi::{
    shared::{basetsd::ULONG_PTR, minwindef::DWORD},
    um::minwinbase::OVERLAPPED,
};

use crate::common::{ANNError, ANNResult};
use crate::model::IOContext;

pub const MAX_IO_CONCURRENCY: usize = 128; // To do: explore the optimal value for this. The current value is taken from C++ code.
pub const FILE_ATTRIBUTE_READONLY: DWORD = 0x00000001;
pub const IO_COMPLETION_TIMEOUT: DWORD = u32::MAX; // Infinite timeout.
pub const DISK_IO_ALIGNMENT: usize = 512;
pub const ASYNC_IO_COMPLETION_CHECK_INTERVAL: Duration = Duration::from_micros(5);

/// Aligned read struct for disk IO, it takes the ownership of the AlignedBoxedSlice and returns the AlignedBoxWithSlice data immutably.  
pub struct AlignedRead<'a, T> {
    /// where to read from
    /// offset needs to be aligned with DISK_IO_ALIGNMENT
    offset: u64,

    /// where to read into
    /// aligned_buf and its len need to be aligned with DISK_IO_ALIGNMENT
    aligned_buf: &'a mut [T],
}

impl<'a, T> AlignedRead<'a, T> {
    pub fn new(offset: u64, aligned_buf: &'a mut [T]) -> ANNResult<Self> {
        Self::assert_is_aligned(offset as usize)?;
        Self::assert_is_aligned(std::mem::size_of_val(aligned_buf))?;

        Ok(Self {
            offset,
            aligned_buf,
        })
    }

    fn assert_is_aligned(val: usize) -> ANNResult<()> {
        match val % DISK_IO_ALIGNMENT {
            0 => Ok(()),
            _ => Err(ANNError::log_disk_io_request_alignment_error(format!(
                "The offset or length of AlignedRead request is not {} bytes aligned",
                DISK_IO_ALIGNMENT
            ))),
        }
    }

    pub fn aligned_buf(&self) -> &[T] {
        self.aligned_buf
    }
}

pub struct WindowsAlignedFileReader {
    file_name: String,

    // ctx_map is the mapping from thread id to io context. It is hashmap behind a sharded lock to allow concurrent access from multiple threads.
    // ShardedLock: shardedlock provides an implementation of a reader-writer lock that offers concurrent read access to the shared data while allowing exclusive write access.
    // It achieves better scalability by dividing the shared data into multiple shards, and each with its own internal lock.
    // Multiple threads can read from different shards simultaneously, reducing contention.
    // https://docs.rs/crossbeam/0.8.2/crossbeam/sync/struct.ShardedLock.html
    // Comparing to RwLock, ShardedLock provides higher concurrency for read operations and is suitable for read heavy workloads.
    // The value of the hashmap is an Arc<IOContext> to allow immutable access to IOContext with automatic reference counting.
    ctx_map: Lazy<ShardedLock<HashMap<thread::ThreadId, Arc<IOContext>>>>,
}

impl WindowsAlignedFileReader {
    pub fn new(fname: &str) -> ANNResult<Self> {
        let reader: WindowsAlignedFileReader = WindowsAlignedFileReader {
            file_name: fname.to_string(),
            ctx_map: Lazy::new(|| ShardedLock::new(HashMap::new())),
        };

        reader.register_thread()?;
        Ok(reader)
    }

    // Register the io context for a thread if it hasn't been registered.
    pub fn register_thread(&self) -> ANNResult<()> {
        let mut ctx_map = self.ctx_map.write().map_err(|_| {
            ANNError::log_lock_poison_error("unable to acquire read lock on ctx_map".to_string())
        })?;

        let id = thread::current().id();
        if ctx_map.contains_key(&id) {
            println!(
                "Warning:: Duplicate registration for thread_id : {:?}. Directly call get_ctx to get the thread context data.", 
                id);

            return Ok(());
        }

        let mut ctx = IOContext::new();

        match unsafe { FileHandle::new(&self.file_name, AccessMode::Read, ShareMode::Read) } {
            Ok(file_handle) => ctx.file_handle = file_handle,
            Err(err) => {
                return Err(ANNError::log_io_error(err));
            }
        }

        // Create a io completion port for the file handle, later it will be used to get the completion status.
        match IOCompletionPort::new(&ctx.file_handle, None, 0, 0) {
            Ok(io_completion_port) => ctx.io_completion_port = io_completion_port,
            Err(err) => {
                return Err(ANNError::log_io_error(err));
            }
        }

        ctx_map.insert(id, Arc::new(ctx));

        Ok(())
    }

    // Get the reference counted io context for the current thread.
    pub fn get_ctx(&self) -> ANNResult<Arc<IOContext>> {
        let ctx_map = self.ctx_map.read().map_err(|_| {
            ANNError::log_lock_poison_error("unable to acquire read lock on ctx_map".to_string())
        })?;

        let id = thread::current().id();
        match ctx_map.get(&id) {
            Some(ctx) => Ok(Arc::clone(ctx)),
            None => Err(ANNError::log_index_error(format!(
                "unable to find IOContext for thread_id {:?}",
                id
            ))),
        }
    }

    // Read the data from the file by sending concurrent io requests in batches.
    pub fn read<T>(&self, read_requests: &mut [AlignedRead<T>], ctx: &IOContext) -> ANNResult<()> {
        let n_requests = read_requests.len();
        let n_batches = (n_requests + MAX_IO_CONCURRENCY - 1) / MAX_IO_CONCURRENCY;

        let mut overlapped_in_out =
            vec![unsafe { std::mem::zeroed::<OVERLAPPED>() }; MAX_IO_CONCURRENCY];

        for batch_idx in 0..n_batches {
            let batch_start = MAX_IO_CONCURRENCY * batch_idx;
            let batch_size = std::cmp::min(n_requests - batch_start, MAX_IO_CONCURRENCY);

            for j in 0..batch_size {
                let req = &mut read_requests[batch_start + j];
                let os = &mut overlapped_in_out[j];

                match unsafe {
                    read_file_to_slice(&ctx.file_handle, req.aligned_buf, os, req.offset)
                } {
                    Ok(_) => {}
                    Err(error) => {
                        return Err(ANNError::IOError { err: (error) });
                    }
                }
            }

            let mut n_read: DWORD = 0;
            let mut n_complete: u64 = 0;
            let mut completion_key: ULONG_PTR = 0;
            let mut lp_os: *mut OVERLAPPED = ptr::null_mut();
            while n_complete < batch_size as u64 {
                match unsafe {
                    get_queued_completion_status(
                        &ctx.io_completion_port,
                        &mut n_read,
                        &mut completion_key,
                        &mut lp_os,
                        IO_COMPLETION_TIMEOUT,
                    )
                } {
                    // An IO request completed.
                    Ok(true) => n_complete += 1,
                    // No IO request completed, continue to wait.
                    Ok(false) => {
                        thread::sleep(ASYNC_IO_COMPLETION_CHECK_INTERVAL);
                    }
                    // An error ocurred.
                    Err(error) => return Err(ANNError::IOError { err: (error) }),
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use bincode::deserialize_from;
    use serde::{Deserialize, Serialize};

    use crate::{common::AlignedBoxWithSlice, model::SECTOR_LEN};

    use super::*;
    pub const TEST_INDEX_PATH: &str =
        "./tests/data/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_alligned_reader_test.index";
    pub const TRUTH_NODE_DATA_PATH: &str =
        "./tests/data/disk_index_node_data_aligned_reader_truth.bin";

    #[derive(Debug, Serialize, Deserialize)]
    struct NodeData {
        num_neighbors: u32,
        coordinates: Vec<f32>,
        neighbors: Vec<u32>,
    }

    impl PartialEq for NodeData {
        fn eq(&self, other: &Self) -> bool {
            self.num_neighbors == other.num_neighbors
                && self.coordinates == other.coordinates
                && self.neighbors == other.neighbors
        }
    }

    #[test]
    fn test_new_aligned_file_reader() {
        // Replace "test_file_path" with actual file path
        let result = WindowsAlignedFileReader::new(TEST_INDEX_PATH);
        assert!(result.is_ok());

        let reader = result.unwrap();
        assert_eq!(reader.file_name, TEST_INDEX_PATH);
    }

    #[test]
    fn test_read() {
        let reader = WindowsAlignedFileReader::new(TEST_INDEX_PATH).unwrap();
        let ctx = reader.get_ctx().unwrap();

        let read_length = 512; // adjust according to your logic
        let num_read = 10;
        let mut aligned_mem = AlignedBoxWithSlice::<u8>::new(read_length * num_read, 512).unwrap();

        // create and add AlignedReads to the vector
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .enumerate()
            .map(|(i, slice)| {
                let offset = (i * read_length) as u64;
                AlignedRead::new(offset, slice).unwrap()
            })
            .collect();

        let result = reader.read(&mut aligned_reads, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_read_disk_index_by_sector() {
        let reader = WindowsAlignedFileReader::new(TEST_INDEX_PATH).unwrap();
        let ctx = reader.get_ctx().unwrap();

        let read_length = SECTOR_LEN; // adjust according to your logic
        let num_sector = 10;
        let mut aligned_mem =
            AlignedBoxWithSlice::<u8>::new(read_length * num_sector, 512).unwrap();

        // Each slice will be used as the buffer for a read request of a sector.
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .enumerate()
            .map(|(sector_id, slice)| {
                let offset = (sector_id * read_length) as u64;
                AlignedRead::new(offset, slice).unwrap()
            })
            .collect();

        let result = reader.read(&mut aligned_reads, &ctx);
        assert!(result.is_ok());

        aligned_reads.iter().for_each(|read| {
            assert_eq!(read.aligned_buf.len(), SECTOR_LEN);
        });

        let disk_layout_meta = reconstruct_disk_meta(aligned_reads[0].aligned_buf);
        assert!(disk_layout_meta.len() > 9);

        let dims = disk_layout_meta[1];
        let num_pts = disk_layout_meta[0];
        let max_node_len = disk_layout_meta[3];
        let max_num_nodes_per_sector = disk_layout_meta[4];

        assert!(max_node_len * max_num_nodes_per_sector < SECTOR_LEN as u64);

        let num_nbrs_start = (dims as usize) * std::mem::size_of::<f32>();
        let nbrs_buf_start = num_nbrs_start + std::mem::size_of::<u32>();

        let mut node_data_array = Vec::with_capacity(max_num_nodes_per_sector as usize * 9);

        // Only validate the first 9 sectors with graph nodes.
        (1..9).for_each(|sector_id| {
            let sector_data = &mem_slices[sector_id];
            for node_data in sector_data.chunks_exact(max_node_len as usize) {
                // Extract coordinates data from the start of the node_data
                let coordinates_end = (dims as usize) * std::mem::size_of::<f32>();
                let coordinates = node_data[0..coordinates_end]
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                // Extract number of neighbors from the node_data
                let neighbors_num = u32::from_le_bytes(
                    node_data[num_nbrs_start..nbrs_buf_start]
                        .try_into()
                        .unwrap(),
                );

                let nbors_buf_end =
                    nbrs_buf_start + (neighbors_num as usize) * std::mem::size_of::<u32>();

                // Extract neighbors from the node data.
                let mut neighbors = Vec::new();
                for nbors_data in node_data[nbrs_buf_start..nbors_buf_end]
                    .chunks_exact(std::mem::size_of::<u32>())
                {
                    let nbors_id = u32::from_le_bytes(nbors_data.try_into().unwrap());
                    assert!(nbors_id < num_pts as u32);
                    neighbors.push(nbors_id);
                }

                // Create NodeData struct and push it to the node_data_array
                node_data_array.push(NodeData {
                    num_neighbors: neighbors_num,
                    coordinates,
                    neighbors,
                });
            }
        });

        // Compare that each node read from the disk index are expected.
        let node_data_truth_file = File::open(TRUTH_NODE_DATA_PATH).unwrap();
        let reader = BufReader::new(node_data_truth_file);

        let node_data_vec: Vec<NodeData> = deserialize_from(reader).unwrap();
        for (node_from_node_data_file, node_from_disk_index) in
            node_data_vec.iter().zip(node_data_array.iter())
        {
            // Verify that the NodeData from the file is equal to the NodeData in node_data_array
            assert_eq!(node_from_node_data_file, node_from_disk_index);
        }
    }

    #[test]
    fn test_read_fail_invalid_file() {
        let reader = WindowsAlignedFileReader::new("/invalid_path");
        assert!(reader.is_err());
    }

    #[test]
    fn test_read_no_requests() {
        let reader = WindowsAlignedFileReader::new(TEST_INDEX_PATH).unwrap();
        let ctx = reader.get_ctx().unwrap();

        let mut read_requests = Vec::<AlignedRead<u8>>::new();
        let result = reader.read(&mut read_requests, &ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_ctx() {
        let reader = WindowsAlignedFileReader::new(TEST_INDEX_PATH).unwrap();
        let result = reader.get_ctx();
        assert!(result.is_ok());
    }

    #[test]
    fn test_register_thread() {
        let reader = WindowsAlignedFileReader::new(TEST_INDEX_PATH).unwrap();
        let result = reader.register_thread();
        assert!(result.is_ok());
    }

    fn reconstruct_disk_meta(buffer: &[u8]) -> Vec<u64> {
        let size_of_u64 = std::mem::size_of::<u64>();

        let num_values = buffer.len() / size_of_u64;
        let mut disk_layout_meta = Vec::with_capacity(num_values);
        let meta_data = &buffer[8..];

        for chunk in meta_data.chunks_exact(size_of_u64) {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            disk_layout_meta.push(value);
        }

        disk_layout_meta
    }
}
