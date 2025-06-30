/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::Mutex;
use num_traits::Num;

/// Non recursive mutex
pub type NonRecursiveMutex = Mutex<()>;

/// Round up X to the nearest multiple of Y
#[inline]
pub fn round_up<T>(x: T, y: T) -> T 
where T : Num + Copy
{
    div_round_up(x, y) * y
}

/// Rounded-up division
#[inline]
pub fn div_round_up<T>(x: T, y: T) -> T 
where T : Num + Copy
{
    (x / y) + if x % y != T::zero() {T::one()} else {T::zero()}
}

/// Round down X to the nearest multiple of Y
#[inline]
pub fn round_down<T>(x: T, y: T) -> T
where T : Num + Copy
{
    (x / y) * y
}

/// Is aligned
#[inline]
pub fn is_aligned<T>(x: T, y: T) -> bool
where T : Num + Copy
{
    x % y == T::zero()
}

#[inline]
pub fn is_512_aligned(x: u64) -> bool {
    is_aligned(x, 512)
}

#[inline]
pub fn is_4096_aligned(x: u64) -> bool {
    is_aligned(x, 4096)
}

/// all metadata of individual sub-component files is written in first 4KB for unified files
pub const METADATA_SIZE: usize = 4096;

pub const BUFFER_SIZE_FOR_CACHED_IO: usize = 1024 * 1048576;

pub const PBSTR: &str = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";

pub const PBWIDTH: usize = 60;

macro_rules! convert_types {
    ($name:ident, $intput_type:ty, $output_type:ty) => {
        /// Write data into file
        pub fn $name(srcmat: &[$intput_type], npts: usize, dim: usize) -> Vec<$output_type> {
            let mut destmat: Vec<$output_type> = Vec::new();
            for i in 0..npts {
                for j in 0..dim {
                    destmat.push(srcmat[i * dim + j] as $output_type);
                }
            }
            destmat
        }
    };
}
convert_types!(convert_types_usize_u8, usize, u8);
convert_types!(convert_types_usize_u32, usize, u32);
convert_types!(convert_types_usize_u64, usize, u64);
convert_types!(convert_types_u64_usize, u64, usize);
convert_types!(convert_types_u32_usize, u32, usize);

#[cfg(test)]
mod file_util_test {
    use super::*;
    use std::any::type_name;

    #[test]
    fn round_up_test() {
        assert_eq!(round_up(252, 8), 256);
        assert_eq!(round_up(256, 8), 256);
    }

    #[test]
    fn div_round_up_test() {
        assert_eq!(div_round_up(252, 8), 32);
        assert_eq!(div_round_up(256, 8), 32);
    }

    #[test]
    fn round_down_test() {
        assert_eq!(round_down(252, 8), 248);
        assert_eq!(round_down(256, 8), 256);
    }

    #[test]
    fn is_aligned_test() {
        assert!(!is_aligned(252, 8));
        assert!(is_aligned(256, 8));
    }

    #[test]
    fn is_512_aligned_test() {
        assert!(!is_512_aligned(520));
        assert!(is_512_aligned(512));
    }

    #[test]
    fn is_4096_aligned_test() {
        assert!(!is_4096_aligned(4090));
        assert!(is_4096_aligned(4096));
    }

    #[test] 
    fn convert_types_test() {
        let data = vec![0u64, 1u64, 2u64];
        let output = convert_types_u64_usize(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "usize");
        assert_eq!(output[0], 0usize);

        let data = vec![0usize, 1usize, 2usize];
        let output = convert_types_usize_u8(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "u8");
        assert_eq!(output[0], 0u8);

        let data = vec![0usize, 1usize, 2usize];
        let output = convert_types_usize_u64(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "u64");
        assert_eq!(output[0], 0u64);

        let data = vec![0u32, 1u32, 2u32];
        let output = convert_types_u32_usize(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "usize");
        assert_eq!(output[0],0usize);
    }

    fn type_of<T>(_: T) -> &'static str {
        type_name::<T>()
    }
}

