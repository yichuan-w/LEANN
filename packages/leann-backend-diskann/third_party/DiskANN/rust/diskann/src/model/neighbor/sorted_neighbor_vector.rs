/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Sorted Neighbor Vector

use std::ops::{Deref, DerefMut};

use super::Neighbor;

/// A newtype on top of vector of neighbors, is sorted by distance
#[derive(Debug)]
pub struct SortedNeighborVector<'a>(&'a mut Vec<Neighbor>);

impl<'a> SortedNeighborVector<'a> {
    /// Create a new SortedNeighborVector
    pub fn new(vec: &'a mut Vec<Neighbor>) -> Self {
        vec.sort_unstable();
        Self(vec)
    }
}

impl<'a> Deref for SortedNeighborVector<'a> {
    type Target = Vec<Neighbor>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> DerefMut for SortedNeighborVector<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}
