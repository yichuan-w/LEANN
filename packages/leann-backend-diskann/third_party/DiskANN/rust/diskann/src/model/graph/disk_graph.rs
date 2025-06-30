/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Disk graph

use byteorder::{LittleEndian, ByteOrder};
use vector::FullPrecisionDistance;

use crate::common::{ANNResult, ANNError};
use crate::model::data_store::DiskScratchDataset;
use crate::model::Vertex;
use crate::storage::DiskGraphStorage;

use super::{VertexAndNeighbors, SectorGraph, AdjacencyList};

/// Disk graph
pub struct DiskGraph {
    /// dim of fp vector in disk sector
    dim: usize,

    /// number of nodes per sector
    num_nodes_per_sector: u64,

    /// max node length in bytes
    max_node_len: u64,

    /// the len of fp vector
    fp_vector_len: u64,

    /// list of nodes (vertex_id) to fetch from disk
    nodes_to_fetch: Vec<u32>,

    /// Sector graph
    sector_graph: SectorGraph,
}

impl<'a> DiskGraph {
    /// Create DiskGraph instance
    pub fn new(
        dim: usize, 
        num_nodes_per_sector: u64,
        max_node_len: u64,
        fp_vector_len: u64,
        beam_width: usize, 
        graph_storage: DiskGraphStorage,
    ) -> ANNResult<Self> {
        let graph = Self {
            dim,
            num_nodes_per_sector,
            max_node_len,
            fp_vector_len,
            nodes_to_fetch: Vec::with_capacity(2 * beam_width),
            sector_graph: SectorGraph::new(graph_storage)?,
        };

        Ok(graph)
    }

    /// Add vertex_id into the list to fetch from disk
    pub fn add_vertex(&mut self, id: u32) {
        self.nodes_to_fetch.push(id);
    }

    /// Fetch nodes from disk index
    pub fn fetch_nodes(&mut self) -> ANNResult<()> {
        let sectors_to_fetch: Vec<u64> = self.nodes_to_fetch.iter().map(|&id| self.node_sector_index(id)).collect();
        self.sector_graph.read_graph(&sectors_to_fetch)?;

        Ok(())
    }

    /// Copy disk fp vector to DiskScratchDataset
    /// Return the fp vector with aligned dim from DiskScratchDataset
    pub fn copy_fp_vector_to_disk_scratch_dataset<T, const N: usize>(
        &self, 
        node_index: usize,
        disk_scratch_dataset: &'a mut DiskScratchDataset<T, N>
    ) -> ANNResult<Vertex<'a, T, N>> 
    where
        [T; N]: FullPrecisionDistance<T, N>,
    {
        if self.dim > N {
            return Err(ANNError::log_index_error(format!(
                "copy_sector_fp_to_aligned_dataset: dim {} is greater than aligned dim {}",
                self.dim, N)));
        }

        let fp_vector_buf = self.node_fp_vector_buf(node_index);

        // Safety condition is met here
        let aligned_dim_vector = unsafe { disk_scratch_dataset.memcpy_from_fp_vector_buf(fp_vector_buf) };

        Vertex::<'a, T, N>::try_from((aligned_dim_vector, self.nodes_to_fetch[node_index]))
            .map_err(|err| ANNError::log_index_error(format!("TryFromSliceError: failed to get Vertex for disk index node, err={}", err)))
    }

    /// Reset graph
    pub fn reset(&mut self) {
        self.nodes_to_fetch.clear();
        self.sector_graph.reset();
    }

    fn get_vertex_and_neighbors(&self, node_index: usize) -> VertexAndNeighbors {
        let node_disk_buf = self.node_disk_buf(node_index);
        let buf = &node_disk_buf[self.fp_vector_len as usize..];
        let num_neighbors = LittleEndian::read_u32(&buf[0..4]) as usize;
        let neighbors_buf = &buf[4..4 + num_neighbors * 4];

        let mut adjacency_list = AdjacencyList::for_range(num_neighbors);
        for chunk in neighbors_buf.chunks(4) {
            let neighbor_id = LittleEndian::read_u32(chunk);
            adjacency_list.push(neighbor_id);
        }

        VertexAndNeighbors::new(self.nodes_to_fetch[node_index], adjacency_list)
    }

    #[inline]
    fn node_sector_index(&self, vertex_id: u32) -> u64 {
        vertex_id as u64 / self.num_nodes_per_sector + 1
    }

    #[inline]
    fn node_disk_buf(&self, node_index: usize) -> &[u8] {
        let vertex_id = self.nodes_to_fetch[node_index];

        // get sector_buf where this node is located
        let sector_buf = self.sector_graph.get_sector_buf(node_index);
        let node_offset = (vertex_id as u64 % self.num_nodes_per_sector * self.max_node_len) as usize;
        &sector_buf[node_offset..node_offset + self.max_node_len as usize]
    }

    #[inline]
    fn node_fp_vector_buf(&self, node_index: usize) -> &[u8] {
        let node_disk_buf = self.node_disk_buf(node_index);
        &node_disk_buf[..self.fp_vector_len as usize]
    }
}

/// Iterator for DiskGraph
pub struct DiskGraphIntoIterator<'a> {
    graph: &'a DiskGraph,
    index: usize,
}

impl<'a> IntoIterator for &'a DiskGraph
{
    type IntoIter = DiskGraphIntoIterator<'a>;
    type Item = ANNResult<(usize, VertexAndNeighbors)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        DiskGraphIntoIterator {
            graph: self,
            index: 0,
        }
    }
}

impl<'a> Iterator for DiskGraphIntoIterator<'a> 
{
    type Item = ANNResult<(usize, VertexAndNeighbors)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.graph.nodes_to_fetch.len() {
            return None;
        }

        let node_index = self.index;    
        let vertex_and_neighbors = self.graph.get_vertex_and_neighbors(self.index);
    
        self.index += 1;
        Some(Ok((node_index, vertex_and_neighbors)))
    }
}

