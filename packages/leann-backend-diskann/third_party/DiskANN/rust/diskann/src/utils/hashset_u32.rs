/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use hashbrown::HashSet;
use std::{hash::BuildHasherDefault, ops::{Deref, DerefMut}};
use fxhash::FxHasher;

lazy_static::lazy_static! {
    /// Singleton hasher.
    static ref HASHER: BuildHasherDefault<FxHasher> = {
        BuildHasherDefault::<FxHasher>::default()
    };
}

pub struct HashSetForU32 {
    hashset: HashSet::<u32, BuildHasherDefault<FxHasher>>,
}

impl HashSetForU32 {
    pub fn with_capacity(capacity: usize) -> HashSetForU32 {
        let hashset = HashSet::<u32, BuildHasherDefault<FxHasher>>::with_capacity_and_hasher(capacity, HASHER.clone());
        HashSetForU32 {
            hashset
        }
    }
}

impl Deref for HashSetForU32 {
    type Target = HashSet::<u32, BuildHasherDefault<FxHasher>>;

    fn deref(&self) -> &Self::Target {
        &self.hashset
    }
}

impl DerefMut for HashSetForU32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.hashset
    }
}

