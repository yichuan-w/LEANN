/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::env;

use diskann::{
    common::{ANNError, ANNResult},
    index::create_inmem_index,
    model::{
        configuration::index_write_parameters::IndexWriteParametersBuilder,
        vertex::{DIM_104, DIM_128, DIM_256},
        IndexConfiguration,
    },
    utils::round_up,
    utils::{file_exists, load_ids_to_delete_from_file, load_metadata_from_file, Timer},
};

use vector::{FullPrecisionDistance, Half, Metric};

// The main function to build an in-memory index
#[allow(clippy::too_many_arguments)]
fn build_and_insert_delete_in_memory_index<T>(
    metric: Metric,
    data_path: &str,
    delta_path: &str,
    r: u32,
    l: u32,
    alpha: f32,
    save_path: &str,
    num_threads: u32,
    _use_pq_build: bool,
    _num_pq_bytes: usize,
    use_opq: bool,
    delete_path: &str,
) -> ANNResult<()>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
{
    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_alpha(alpha)
        .with_saturate_graph(false)
        .with_num_threads(num_threads)
        .build();

    let (data_num, data_dim) = load_metadata_from_file(data_path)?;

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        data_num,
        false,
        0,
        use_opq,
        0,
        2.0f32,
        index_write_parameters,
    );
    let mut index = create_inmem_index::<T>(config)?;

    let timer = Timer::new();

    index.build(data_path, data_num)?;

    let diff = timer.elapsed();

    println!("Initial indexing time: {}", diff.as_secs_f64());

    let (delta_data_num, _) = load_metadata_from_file(delta_path)?;

    index.insert(delta_path, delta_data_num)?;

    if !delete_path.is_empty() {
        if !file_exists(delete_path) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file for delete {} does not exist.",
                delete_path
            )));
        }

        let (num_points_to_delete, vertex_ids_to_delete) =
            load_ids_to_delete_from_file(delete_path)?;
        index.soft_delete(vertex_ids_to_delete, num_points_to_delete)?;
    }

    index.save(save_path)?;

    Ok(())
}

fn main() -> ANNResult<()> {
    let mut data_type = String::new();
    let mut dist_fn = String::new();
    let mut data_path = String::new();
    let mut insert_path = String::new();
    let mut index_path_prefix = String::new();
    let mut delete_path = String::new();

    let mut num_threads = 0u32;
    let mut r = 64u32;
    let mut l = 100u32;

    let mut alpha = 1.2f32;
    let mut build_pq_bytes = 0u32;
    let mut _use_pq_build = false;
    let mut use_opq = false;

    let args: Vec<String> = env::args().collect();
    let mut iter = args.iter().skip(1).peekable();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--data_type" => {
                data_type = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "data_type".to_string(),
                            "Missing data type".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--dist_fn" => {
                dist_fn = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "dist_fn".to_string(),
                            "Missing distance function".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--data_path" => {
                data_path = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "data_path".to_string(),
                            "Missing data path".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--insert_path" => {
                insert_path = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "insert_path".to_string(),
                            "Missing insert path".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--index_path_prefix" => {
                index_path_prefix = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "index_path_prefix".to_string(),
                            "Missing index path prefix".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--max_degree" | "-R" => {
                r = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "max_degree".to_string(),
                            "Missing max degree".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "max_degree".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--Lbuild" | "-L" => {
                l = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "Lbuild".to_string(),
                            "Missing build complexity".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "Lbuild".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--alpha" => {
                alpha = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "alpha".to_string(),
                            "Missing alpha".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "alpha".to_string(),
                            format!("ParseFloatError: {}", err),
                        )
                    })?;
            }
            "--num_threads" | "-T" => {
                num_threads = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "num_threads".to_string(),
                            "Missing number of threads".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "num_threads".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--build_PQ_bytes" => {
                build_pq_bytes = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "build_PQ_bytes".to_string(),
                            "Missing PQ bytes".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "build_PQ_bytes".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--use_opq" => {
                use_opq = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "use_opq".to_string(),
                            "Missing use_opq flag".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "use_opq".to_string(),
                            format!("ParseBoolError: {}", err),
                        )
                    })?;
            }
            "--delete_path" => {
                delete_path = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "delete_path".to_string(),
                            "Missing delete_path".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "delete_set_path".to_string(),
                            format!("ParseStringError: {}", err),
                        )
                    })?;
            }
            _ => {
                return Err(ANNError::log_index_config_error(
                    String::from(""),
                    format!("Unknown argument: {}", arg),
                ));
            }
        }
    }

    if data_type.is_empty()
        || dist_fn.is_empty()
        || data_path.is_empty()
        || index_path_prefix.is_empty()
    {
        return Err(ANNError::log_index_config_error(
            String::from(""),
            "Missing required arguments".to_string(),
        ));
    }

    _use_pq_build = build_pq_bytes > 0;

    let metric = dist_fn
        .parse::<Metric>()
        .map_err(|err| ANNError::log_index_config_error("dist_fn".to_string(), err.to_string()))?;

    println!(
        "Starting index build with R: {}  Lbuild: {}  alpha: {}  #threads: {}",
        r, l, alpha, num_threads
    );

    match data_type.as_str() {
        "int8" => {
            build_and_insert_delete_in_memory_index::<i8>(
                metric,
                &data_path,
                &insert_path,
                r,
                l,
                alpha,
                &index_path_prefix,
                num_threads,
                _use_pq_build,
                build_pq_bytes as usize,
                use_opq,
                &delete_path,
            )?;
        }
        "uint8" => {
            build_and_insert_delete_in_memory_index::<u8>(
                metric,
                &data_path,
                &insert_path,
                r,
                l,
                alpha,
                &index_path_prefix,
                num_threads,
                _use_pq_build,
                build_pq_bytes as usize,
                use_opq,
                &delete_path,
            )?;
        }
        "float" => {
            build_and_insert_delete_in_memory_index::<f32>(
                metric,
                &data_path,
                &insert_path,
                r,
                l,
                alpha,
                &index_path_prefix,
                num_threads,
                _use_pq_build,
                build_pq_bytes as usize,
                use_opq,
                &delete_path,
            )?;
        }
        "f16" => {
            build_and_insert_delete_in_memory_index::<Half>(
                metric,
                &data_path,
                &insert_path,
                r,
                l,
                alpha,
                &index_path_prefix,
                num_threads,
                _use_pq_build,
                build_pq_bytes as usize,
                use_opq,
                &delete_path,
            )?;
        }
        _ => {
            println!("Unsupported type. Use one of int8, uint8 or float.");
            return Err(ANNError::log_index_config_error(
                "data_type".to_string(),
                "Invalid data type".to_string(),
            ));
        }
    }

    Ok(())
}

fn print_help() {
    println!("Arguments");
    println!("--help, -h                Print information on arguments");
    println!("--data_type               data type <int8/uint8/float> (required)");
    println!("--dist_fn                 distance function <l2/cosine> (required)");
    println!(
        "--data_path               Input data file in bin format for initial build (required)"
    );
    println!("--insert_path             Input data file in bin format for insert (required)");
    println!("--index_path_prefix       Path prefix for saving index file components (required)");
    println!("--max_degree, -R          Maximum graph degree (default: 64)");
    println!("--Lbuild, -L              Build complexity, higher value results in better graphs (default: 100)");
    println!("--alpha                   alpha controls density and diameter of graph, set 1 for sparse graph, 1.2 or 1.4 for denser graphs with lower diameter (default: 1.2)");
    println!("--num_threads, -T         Number of threads used for building index (defaults to num of CPU logic cores)");
    println!("--build_PQ_bytes          Number of PQ bytes to build the index; 0 for full precision build (default: 0)");
    println!("--use_opq                 Set true for OPQ compression while using PQ distance comparisons for building the index, and false for PQ compression (default: false)");
}

