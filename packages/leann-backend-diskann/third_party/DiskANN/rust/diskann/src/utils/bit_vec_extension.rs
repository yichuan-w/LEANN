/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::cmp::Ordering;

use bit_vec::BitVec;

pub trait BitVecExtension {
    fn resize(&mut self, new_len: usize, value: bool);
}

impl BitVecExtension for BitVec {
    fn resize(&mut self, new_len: usize, value: bool) {
        let old_len = self.len();
        match new_len.cmp(&old_len) {
            Ordering::Less => self.truncate(new_len),
            Ordering::Greater => self.grow(new_len - old_len, value),
            Ordering::Equal => {}
        }
    }
}

#[cfg(test)]
mod bit_vec_extension_test {
    use super::*;

    #[test]
    fn resize_test() {
        let mut bitset = BitVec::new();

        bitset.resize(10, false);
        assert_eq!(bitset.len(), 10);
        assert!(bitset.none());

        bitset.resize(11, true);
        assert_eq!(bitset.len(), 11);
        assert!(bitset[10]);

        bitset.resize(5, false);
        assert_eq!(bitset.len(), 5);
        assert!(bitset.none());
    }
}

