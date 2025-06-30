/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use vector::Metric;

use crate::index::InmemIndex;
use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use crate::model::{IndexConfiguration};
use crate::model::vertex::DIM_128;
use crate::utils::{file_exists, load_metadata_from_file};

use super::get_test_file_path;

// f32, 128 DIM and 256 points source data
const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
const NUM_POINTS_TO_LOAD: usize = 256;

pub fn create_index_with_test_data() -> InmemIndex<f32, DIM_128> {
    let index_write_parameters = IndexWriteParametersBuilder::new(50, 4).with_alpha(1.2).build();
    let config = IndexConfiguration::new(
        Metric::L2, 
        128, 
        128,
        256, 
        false, 
        0, 
        false,
        0, 
        1.0f32,
        index_write_parameters);
    let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config).unwrap();

    build_test_index(&mut index, get_test_file_path(TEST_DATA_FILE).as_str(), NUM_POINTS_TO_LOAD);

    index.start = index.dataset.calculate_medoid_point_id().unwrap();

    index
}

fn build_test_index(index: &mut InmemIndex<f32, DIM_128>, filename: &str, num_points_to_load: usize) {
    if !file_exists(filename) {
        panic!("ERROR: Data file {} does not exist.", filename);
    }

    let (file_num_points, file_dim) = load_metadata_from_file(filename).unwrap();
    if file_num_points > index.configuration.max_points {
        panic!(
            "ERROR: Driver requests loading {} points and file has {} points, 
        but index can support only {} points as specified in configuration.",
            num_points_to_load, file_num_points, index.configuration.max_points
        );
    }

    if num_points_to_load > file_num_points {
        panic!(
            "ERROR: Driver requests loading {} points and file has only {} points.",
            num_points_to_load, file_num_points
        );
    }

    if file_dim != index.configuration.dim {
        panic!(
            "ERROR: Driver requests loading {}  dimension, but file has {} dimension.",
            index.configuration.dim, file_dim
        );
    }

    index.dataset.build_from_file(filename, num_points_to_load).unwrap();

    println!("Using only first {} from file.", num_points_to_load);

    index.num_active_pts = num_points_to_load;
}
