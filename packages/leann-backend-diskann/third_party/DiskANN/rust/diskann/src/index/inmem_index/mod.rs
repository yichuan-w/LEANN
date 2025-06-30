/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[allow(clippy::module_inception)]
mod inmem_index;
pub use inmem_index::InmemIndex;

mod inmem_index_storage;

pub mod ann_inmem_index;

