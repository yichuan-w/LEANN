/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Disk graph storage

use std::sync::Arc;

use crate::{model::{WindowsAlignedFileReader, IOContext, AlignedRead}, common::ANNResult};

/// Graph storage for disk index
/// One thread has one storage instance
pub struct DiskGraphStorage {
    /// Disk graph reader
    disk_graph_reader: Arc<WindowsAlignedFileReader>,

    /// IOContext of current thread
    ctx: Arc<IOContext>,
}

impl DiskGraphStorage {
    /// Create a new DiskGraphStorage instance
    pub fn new(disk_graph_reader: Arc<WindowsAlignedFileReader>) -> ANNResult<Self> {
        let ctx = disk_graph_reader.get_ctx()?;
        Ok(Self {
            disk_graph_reader,
            ctx,
        })
    }

    /// Read disk graph data
    pub fn read<T>(&self, read_requests: &mut [AlignedRead<T>]) -> ANNResult<()> {
        self.disk_graph_reader.read(read_requests, &self.ctx)
    }
}
