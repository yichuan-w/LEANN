/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Adjacency List

use std::ops::{Deref, DerefMut};

#[derive(Debug, Eq, PartialEq)]
/// Represents the out neighbors of a vertex
pub struct AdjacencyList {
    edges: Vec<u32>,
}

/// In-mem index related limits
const GRAPH_SLACK_FACTOR: f32 = 1.3_f32;

impl AdjacencyList {
    /// Create AdjacencyList with capacity slack for a range.
    pub fn for_range(range: usize) -> Self {
        let capacity = (range as f32 * GRAPH_SLACK_FACTOR).ceil() as usize;
        Self {
            edges: Vec::with_capacity(capacity),
        }
    }

    /// Push a node to the list of neighbors for the given node.
    pub fn push(&mut self, node_id: u32) {
        debug_assert!(self.edges.len() < self.edges.capacity());
        self.edges.push(node_id);
    }
}

impl From<Vec<u32>> for AdjacencyList {
    fn from(edges: Vec<u32>) -> Self {
        Self { edges }
    }
}

impl Deref for AdjacencyList {
    type Target = Vec<u32>;

    fn deref(&self) -> &Self::Target {
        &self.edges
    }
}

impl DerefMut for AdjacencyList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.edges
    }
}

impl<'a> IntoIterator for &'a AdjacencyList {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

