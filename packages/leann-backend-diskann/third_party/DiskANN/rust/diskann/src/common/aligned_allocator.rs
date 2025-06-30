/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::alloc::Layout;
use std::ops::{Deref, DerefMut, Range};
use std::ptr::copy_nonoverlapping;

use super::{ANNResult, ANNError};

#[derive(Debug)]
/// A box that holds a slice but is aligned to the specified layout.
///
/// This type is useful for working with types that require a certain alignment,
/// such as SIMD vectors or FFI structs. It allocates memory using the global allocator
/// and frees it when dropped. It also implements Deref and DerefMut to allow access
/// to the underlying slice.
pub struct AlignedBoxWithSlice<T> {
    /// The layout of the allocated memory.
    layout: Layout,

    /// The slice that points to the allocated memory.
    val: Box<[T]>,
}

impl<T> AlignedBoxWithSlice<T> {
    /// Creates a new `AlignedBoxWithSlice` with the given capacity and alignment.
    /// The allocated memory are set to 0.
    ///
    /// # Error
    ///
    /// Return IndexError if the alignment is not a power of two or if the layout is invalid.
    ///
    /// This function is unsafe because it allocates uninitialized memory and casts it to
    /// a slice of `T`. The caller must ensure that the capacity and alignment are valid
    /// for the type `T` and that the memory is initialized before accessing the elements
    /// of the slice.
    pub fn new(capacity: usize, alignment: usize) -> ANNResult<Self> {
        let allocsize = capacity.checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ANNError::log_index_error("capacity overflow".to_string()))?;
        let layout = Layout::from_size_align(allocsize, alignment)
            .map_err(ANNError::log_mem_alloc_layout_error)?;

        let val = unsafe {
            let mem = std::alloc::alloc_zeroed(layout);
            let ptr = mem as *mut T;
            let slice = std::slice::from_raw_parts_mut(ptr, capacity);
            std::boxed::Box::from_raw(slice)
        };

        Ok(Self { layout, val })
    }

    /// Returns a reference to the slice.
    pub fn as_slice(&self) -> &[T] {
        &self.val
    }

    /// Returns a mutable reference to the slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.val
    }

    /// Copies data from the source slice to the destination box.
    pub fn memcpy(&mut self, src: &[T]) -> ANNResult<()> {
        if src.len() > self.val.len() {
            return Err(ANNError::log_index_error(format!("source slice is too large (src:{}, dst:{})", src.len(), self.val.len())));
        }

        // Check that they don't overlap
        let src_ptr = src.as_ptr();
        let src_end = unsafe { src_ptr.add(src.len()) };
        let dst_ptr = self.val.as_mut_ptr();
        let dst_end = unsafe { dst_ptr.add(self.val.len()) };

        if src_ptr < dst_end && src_end > dst_ptr {
            return Err(ANNError::log_index_error("Source and destination overlap".to_string()));
        }

        unsafe {
            copy_nonoverlapping(src.as_ptr(), self.val.as_mut_ptr(), src.len());
        }

        Ok(())
    }

    /// Split the range of memory into nonoverlapping mutable slices.
    /// The number of returned slices is (range length / slice_len) and each has a length of slice_len.
    pub fn split_into_nonoverlapping_mut_slices(&mut self, range: Range<usize>, slice_len: usize) -> ANNResult<Vec<&mut [T]>> {
        if range.len() % slice_len != 0 || range.end > self.len() {
            return Err(ANNError::log_index_error(format!(
                "Cannot split range ({:?}) of AlignedBoxWithSlice (len: {}) into nonoverlapping mutable slices with length {}", 
                range,
                self.len(), 
                slice_len,
            )));
        }

        let mut slices = Vec::with_capacity(range.len() / slice_len);
        let mut remaining_slice = &mut self.val[range];

        while remaining_slice.len() >= slice_len {
            let (left, right) = remaining_slice.split_at_mut(slice_len);
            slices.push(left);
            remaining_slice = right;
        }

        Ok(slices)
    }
}


impl<T> Drop for AlignedBoxWithSlice<T> {
    /// Frees the memory allocated for the slice using the global allocator.
    fn drop(&mut self) {
        let val = std::mem::take(&mut self.val);
        let mut val2 = std::mem::ManuallyDrop::new(val);
        let ptr = val2.as_mut_ptr();

        unsafe {
            // let nonNull = NonNull::new_unchecked(ptr as *mut u8);
            std::alloc::dealloc(ptr as *mut u8, self.layout)
        }
    }
}

impl<T> Deref for AlignedBoxWithSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> DerefMut for AlignedBoxWithSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::utils::is_aligned;

    use super::*;

    #[test]
    fn create_alignedvec_works_32() {
        (0..100).for_each(|_| {
            let size = 1_000_000;
            println!("Attempting {}", size);
            let data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
            assert_eq!(data.len(), size, "Capacity should match");

            let ptr = data.as_ptr() as usize;
            assert_eq!(ptr % 32, 0, "Ptr should be aligned to 32");

            // assert that the slice is initialized.
            (0..size).for_each(|i| {
                assert_eq!(data[i], f32::default());
            });

            drop(data);
        });
    }

    #[test]
    fn create_alignedvec_works_256() {
        let mut rng = rand::thread_rng();

        (0..100).for_each(|_| {
            let n = rng.gen::<u8>();
            let size = usize::from(n) + 1;
            println!("Attempting {}", size);
            let data = AlignedBoxWithSlice::<u8>::new(size, 256).unwrap();
            assert_eq!(data.len(), size, "Capacity should match");

            let ptr = data.as_ptr() as usize;
            assert_eq!(ptr % 256, 0, "Ptr should be aligned to 32");

            // assert that the slice is initialized.
            (0..size).for_each(|i| {
                assert_eq!(data[i], u8::default());
            });

            drop(data);
        });
    }

    #[test]
    fn as_slice_test() {
        let size = 1_000_000;
        let data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        // assert that the slice is initialized.
        (0..size).for_each(|i| {
            assert_eq!(data[i], f32::default());
        });

        let slice = data.as_slice();
        (0..size).for_each(|i| {
            assert_eq!(slice[i], f32::default());
        });
    }

    #[test]
    fn as_mut_slice_test() {
        let size = 1_000_000;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut_slice = data.as_mut_slice();
        (0..size).for_each(|i| {
            assert_eq!(mut_slice[i], f32::default());
        });
    }

    #[test]
    fn memcpy_test() {
        let size = 1_000_000;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut destination = AlignedBoxWithSlice::<f32>::new(size-2, 32).unwrap();
        let mut_destination = destination.as_mut_slice();
        data.memcpy(mut_destination).unwrap();
        (0..size-2).for_each(|i| {
            assert_eq!(data[i], mut_destination[i]);
        });
    }

    #[test]
    #[should_panic(expected = "source slice is too large (src:1000000, dst:999998)")]
    fn memcpy_panic_test() {
        let size = 1_000_000;
        let mut data = AlignedBoxWithSlice::<f32>::new(size-2, 32).unwrap();
        let mut destination = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut_destination = destination.as_mut_slice();
        data.memcpy(mut_destination).unwrap();
    }

    #[test]
    fn is_aligned_test() {
        assert!(is_aligned(256,256));
        assert!(!is_aligned(255,256));
    }

    #[test]
    fn split_into_nonoverlapping_mut_slices_test() {
        let size = 10;
        let slice_len = 2;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let slices = data.split_into_nonoverlapping_mut_slices(2..8, slice_len).unwrap();
        assert_eq!(slices.len(), 3);
        for (i, slice) in slices.into_iter().enumerate() {
            assert_eq!(slice.len(), slice_len);
            slice[0] = i as f32 + 1.0;
            slice[1] = i as f32 + 1.0;
        }
        let expected_arr = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0];
        assert_eq!(data.as_ref(), &expected_arr);
    }

    #[test]
    fn split_into_nonoverlapping_mut_slices_error_when_indivisible() {
        let size = 10;
        let slice_len = 2;
        let range = 2..7;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let result = data.split_into_nonoverlapping_mut_slices(range.clone(), slice_len);
        let expected_err_str = format!(
            "IndexError: Cannot split range ({:?}) of AlignedBoxWithSlice (len: {}) into nonoverlapping mutable slices with length {}", 
            range,
            size, 
            slice_len,
        );
        assert!(result.is_err_and(|e| e.to_string() == expected_err_str));
    }
}

