/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[allow(clippy::module_inception)]
mod inmem_dataset;
pub use inmem_dataset::InmemDataset;
pub use inmem_dataset::DatasetDto;

mod disk_scratch_dataset;
pub use disk_scratch_dataset::*;
