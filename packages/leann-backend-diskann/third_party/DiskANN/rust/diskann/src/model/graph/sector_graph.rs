/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Sector graph

use std::ops::Deref;

use crate::common::{AlignedBoxWithSlice, ANNResult, ANNError};
use crate::model::{MAX_N_SECTOR_READS, SECTOR_LEN, AlignedRead};
use crate::storage::DiskGraphStorage;

/// Sector graph read from disk index
pub struct SectorGraph {
    /// Sector bytes from disk
    /// One sector has num_nodes_per_sector nodes
    /// Each node's layout: {full precision vector:[T; DIM]}{num_nbrs: u32}{neighbors: [u32; num_nbrs]}
    /// The fp vector is not aligned
    sectors_data: AlignedBoxWithSlice<u8>,

    /// Graph storage to read sectors
    graph_storage: DiskGraphStorage,

    /// Current sector index into which the next read reads data
    cur_sector_idx: u64,
}

impl SectorGraph {
    /// Create SectorGraph instance
    pub fn new(graph_storage: DiskGraphStorage) -> ANNResult<Self> {
        Ok(Self { 
            sectors_data: AlignedBoxWithSlice::new(MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN)?, 
            graph_storage,
            cur_sector_idx: 0,
        })
    }

    /// Reset SectorGraph
    pub fn reset(&mut self) {
        self.cur_sector_idx = 0;
    }

    /// Read sectors into sectors_data
    /// They are in the same order as sectors_to_fetch
    pub fn read_graph(&mut self, sectors_to_fetch: &[u64]) -> ANNResult<()> {
        let cur_sector_idx_usize: usize = self.cur_sector_idx.try_into()?;
        if sectors_to_fetch.len() > MAX_N_SECTOR_READS - cur_sector_idx_usize {
            return Err(ANNError::log_index_error(format!(
                "Trying to read too many sectors. number of sectors to read: {}, max number of sectors can read: {}", 
                sectors_to_fetch.len(), 
                MAX_N_SECTOR_READS - cur_sector_idx_usize,
            )));
        }

        let mut sector_slices = self.sectors_data.split_into_nonoverlapping_mut_slices(
            cur_sector_idx_usize * SECTOR_LEN..(cur_sector_idx_usize + sectors_to_fetch.len()) * SECTOR_LEN, 
            SECTOR_LEN)?;

        let mut read_requests = Vec::with_capacity(sector_slices.len());
        for (local_sector_idx, slice) in sector_slices.iter_mut().enumerate() {
            let sector_id = sectors_to_fetch[local_sector_idx];
            read_requests.push(AlignedRead::new(sector_id * SECTOR_LEN as u64, slice)?);
        }

        self.graph_storage.read(&mut read_requests)?;
        self.cur_sector_idx += sectors_to_fetch.len() as u64;

        Ok(())
    }

    /// Get sector data by local index
    #[inline]
    pub fn get_sector_buf(&self, local_sector_idx: usize) -> &[u8] {
        &self.sectors_data[local_sector_idx * SECTOR_LEN..(local_sector_idx + 1) * SECTOR_LEN]
    }
}

impl Deref for SectorGraph {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.sectors_data
    }
}

