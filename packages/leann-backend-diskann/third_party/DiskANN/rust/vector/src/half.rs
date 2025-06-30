/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use bytemuck::{Pod, Zeroable};
use half::f16;
use std::convert::AsRef;
use std::fmt;

// Define the Half type as a new type over f16.
// the memory layout of the Half struct will be the same as the memory layout of the f16 type itself.
// The Half struct serves as a simple wrapper around the f16 type and does not introduce any additional memory overhead.
// Test function:
// use half::f16;
// pub struct Half(f16);
// fn main() {
//     let size_of_half = std::mem::size_of::<Half>();
//     let alignment_of_half = std::mem::align_of::<Half>();
//     println!("Size of Half: {} bytes", size_of_half);
//     println!("Alignment of Half: {} bytes", alignment_of_half);
// }
// Output:
// Size of Half: 2 bytes
// Alignment of Half: 2 bytes
pub struct Half(f16);

unsafe impl Pod for Half {}
unsafe impl Zeroable for Half {}

// Implement From<f32> for Half
impl From<Half> for f32 {
    fn from(val: Half) -> Self {
        val.0.to_f32()
    }
}

// Implement AsRef<f16> for Half so that it can be used in distance_compare.
impl AsRef<f16> for Half {
    fn as_ref(&self) -> &f16 {
        &self.0
    }
}

// Implement From<f32> for Half.
impl Half {
    pub fn from_f32(value: f32) -> Self {
        Self(f16::from_f32(value))
    }
}

// Implement Default for Half.
impl Default for Half {
    fn default() -> Self {
        Self(f16::from_f32(Default::default()))
    }
}

// Implement Clone for Half.
impl Clone for Half {
    fn clone(&self) -> Self {
        Half(self.0)
    }
}

// Implement PartialEq for Half.
impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({:?})", self.0)
    }
}

impl Copy for Half {}

impl Half {
    pub fn to_f32(&self) -> f32 {
        self.0.to_f32()
    }
}

unsafe impl Send for Half {}
unsafe impl Sync for Half {}

