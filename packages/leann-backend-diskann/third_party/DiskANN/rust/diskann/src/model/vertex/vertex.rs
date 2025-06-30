/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Vertex

use std::array::TryFromSliceError;

use vector::{FullPrecisionDistance, Metric};

/// Vertex with data type T and dimension N
#[derive(Debug)]
pub struct Vertex<'a, T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Vertex value
    val: &'a [T; N],

    /// Vertex Id
    id: u32,
}

impl<'a, T, const N: usize> Vertex<'a, T, N>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create the vertex with data
    pub fn new(val: &'a [T; N], id: u32) -> Self {
        Self {
            val,
            id,
        }
    }

    /// Compare the vertex with another.
    #[inline(always)]
    pub fn compare(&self, other: &Vertex<'a, T, N>, metric: Metric) -> f32 {
        <[T; N]>::distance_compare(self.val, other.val, metric)
    }

    /// Get the vector associated with the vertex.
    #[inline]
    pub fn vector(&self) -> &[T; N] {
        self.val
    }

    /// Get the vertex id.
    #[inline]
    pub fn vertex_id(&self) -> u32 {
        self.id
    }
}

impl<'a, T, const N: usize> TryFrom<(&'a [T], u32)> for Vertex<'a, T, N>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    type Error = TryFromSliceError;

    fn try_from((mem_slice, id): (&'a [T], u32)) -> Result<Self, Self::Error> {
        let array: &[T; N] = mem_slice.try_into()?;
        Ok(Vertex::new(array, id))
    }
}

