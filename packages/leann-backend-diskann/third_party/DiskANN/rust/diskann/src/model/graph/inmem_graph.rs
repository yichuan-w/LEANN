/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! In-memory graph

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::common::ANNError;

use super::VertexAndNeighbors;

/// The entire graph of in-memory index
#[derive(Debug)]
pub struct InMemoryGraph {
    /// The entire graph
    pub final_graph: Vec<RwLock<VertexAndNeighbors>>,
}

impl InMemoryGraph {
    /// Create InMemoryGraph instance
    pub fn new(size: usize, max_degree: u32) -> Self {
        let mut graph = Vec::with_capacity(size);
        for id in 0..size {
            graph.push(RwLock::new(VertexAndNeighbors::for_range(
                id as u32,
                max_degree as usize,
            )));
        }
        Self { final_graph: graph }
    }

    /// Size of graph
    pub fn size(&self) -> usize {
        self.final_graph.len()
    }

    /// Extend the graph by size vectors
    pub fn extend(&mut self, size: usize, max_degree: u32) {
        for id in 0..size {
            self.final_graph
                .push(RwLock::new(VertexAndNeighbors::for_range(
                    id as u32,
                    max_degree as usize,
                )));
        }
    }

    /// Get read guard of vertex_id
    pub fn read_vertex_and_neighbors(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockReadGuard<VertexAndNeighbors>, ANNError> {
        self.final_graph[vertex_id as usize].read().map_err(|err| {
            ANNError::log_lock_poison_error(format!(
                "PoisonError: Lock poisoned when reading final_graph for vertex_id {}, err={}",
                vertex_id, err
            ))
        })
    }

    /// Get write guard of vertex_id
    pub fn write_vertex_and_neighbors(
        &self,
        vertex_id: u32,
    ) -> Result<RwLockWriteGuard<VertexAndNeighbors>, ANNError> {
        self.final_graph[vertex_id as usize].write().map_err(|err| {
            ANNError::log_lock_poison_error(format!(
                "PoisonError: Lock poisoned when writing final_graph for vertex_id {}, err={}",
                vertex_id, err
            ))
        })
    }
}

#[cfg(test)]
mod graph_tests {
    use crate::model::{graph::AdjacencyList, GRAPH_SLACK_FACTOR};

    use super::*;

    #[test]
    fn test_new() {
        let graph = InMemoryGraph::new(10, 10);
        let capacity = (GRAPH_SLACK_FACTOR * 10_f64).ceil() as usize;

        assert_eq!(graph.final_graph.len(), 10);
        for i in 0..10 {
            let neighbor = graph.final_graph[i].read().unwrap();
            assert_eq!(neighbor.vertex_id, i as u32);
            assert_eq!(neighbor.get_neighbors().capacity(), capacity);
        }
    }

    #[test]
    fn test_size() {
        let graph = InMemoryGraph::new(10, 10);
        assert_eq!(graph.size(), 10);
    }

    #[test]
    fn test_extend() {
        let mut graph = InMemoryGraph::new(10, 10);
        graph.extend(10, 10);

        assert_eq!(graph.size(), 20);

        let capacity = (GRAPH_SLACK_FACTOR * 10_f64).ceil() as usize;
        let mut id: u32 = 0;

        for i in 10..20 {
            let neighbor = graph.final_graph[i].read().unwrap();
            assert_eq!(neighbor.vertex_id, id);
            assert_eq!(neighbor.get_neighbors().capacity(), capacity);
            id += 1;
        }
    }

    #[test]
    fn test_read_vertex_and_neighbors() {
        let graph = InMemoryGraph::new(10, 10);
        let neighbor = graph.read_vertex_and_neighbors(0);
        assert!(neighbor.is_ok());
        assert_eq!(neighbor.unwrap().vertex_id, 0);
    }

    #[test]
    fn test_write_vertex_and_neighbors() {
        let graph = InMemoryGraph::new(10, 10);
        {
            let neighbor = graph.write_vertex_and_neighbors(0);
            assert!(neighbor.is_ok());
            neighbor.unwrap().add_to_neighbors(10, 10);
        }

        let neighbor = graph.read_vertex_and_neighbors(0).unwrap();
        assert_eq!(neighbor.get_neighbors(), &AdjacencyList::from(vec![10_u32]));
    }
}
