/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::env;

use diskann::{
    common::{ANNError, ANNResult},
    index::ann_disk_index::create_disk_index,
    model::{
        default_param_vals::ALPHA,
        vertex::{DIM_104, DIM_128, DIM_256},
        DiskIndexBuildParameters, IndexConfiguration, IndexWriteParametersBuilder,
    },
    storage::DiskIndexStorage,
    utils::round_up,
    utils::{load_metadata_from_file, Timer},
};

use vector::{FullPrecisionDistance, Half, Metric};

/// The main function to build a disk index
#[allow(clippy::too_many_arguments)]
fn build_disk_index<T>(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    search_ram_limit_gb: f64,
    index_build_ram_limit_gb: f64,
    num_pq_chunks: usize,
    use_opq: bool,
) -> ANNResult<()>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
{
    let disk_index_build_parameters =
        DiskIndexBuildParameters::new(search_ram_limit_gb, index_build_ram_limit_gb)?;

    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_saturate_graph(true)
        .with_num_threads(num_threads)
        .build();

    let (data_num, data_dim) = load_metadata_from_file(data_path)?;

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        data_num,
        num_pq_chunks > 0,
        num_pq_chunks,
        use_opq,
        0,
        1f32,
        index_write_parameters,
    );
    let storage = DiskIndexStorage::new(data_path.to_string(), index_path_prefix.to_string())?;
    let mut index = create_disk_index::<T>(Some(disk_index_build_parameters), config, storage)?;

    let timer = Timer::new();

    index.build("")?;

    let diff = timer.elapsed();
    println!("Indexing time: {}", diff.as_secs_f64());

    Ok(())
}

fn main() -> ANNResult<()> {
    let mut data_type = String::new();
    let mut dist_fn = String::new();
    let mut data_path = String::new();
    let mut index_path_prefix = String::new();

    let mut num_threads = 0u32;
    let mut r = 64u32;
    let mut l = 100u32;
    let mut search_ram_limit_gb = 0f64;
    let mut index_build_ram_limit_gb = 0f64;

    let mut build_pq_bytes = 0u32;
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
            "--search_DRAM_budget" | "-B" => {
                search_ram_limit_gb = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "search_DRAM_budget".to_string(),
                            "Missing search_DRAM_budget flag".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "search_DRAM_budget".to_string(),
                            format!("ParseBoolError: {}", err),
                        )
                    })?;
            }
            "--build_DRAM_budget" | "-M" => {
                index_build_ram_limit_gb = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "build_DRAM_budget".to_string(),
                            "Missing build_DRAM_budget flag".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "build_DRAM_budget".to_string(),
                            format!("ParseBoolError: {}", err),
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

    let metric = dist_fn
        .parse::<Metric>()
        .map_err(|err| ANNError::log_index_config_error("dist_fn".to_string(), err.to_string()))?;

    println!(
        "Starting index build with R: {}  Lbuild: {}  alpha: {}  #threads: {} search_DRAM_budget: {} build_DRAM_budget: {}",
        r, l, ALPHA, num_threads, search_ram_limit_gb, index_build_ram_limit_gb
    );

    let err = match data_type.as_str() {
        "int8" => build_disk_index::<i8>(
            metric,
            &data_path,
            r,
            l,
            &index_path_prefix,
            num_threads,
            search_ram_limit_gb,
            index_build_ram_limit_gb,
            build_pq_bytes as usize,
            use_opq,
        ),
        "uint8" => build_disk_index::<u8>(
            metric,
            &data_path,
            r,
            l,
            &index_path_prefix,
            num_threads,
            search_ram_limit_gb,
            index_build_ram_limit_gb,
            build_pq_bytes as usize,
            use_opq,
        ),
        "float" => build_disk_index::<f32>(
            metric,
            &data_path,
            r,
            l,
            &index_path_prefix,
            num_threads,
            search_ram_limit_gb,
            index_build_ram_limit_gb,
            build_pq_bytes as usize,
            use_opq,
        ),
        "f16" => build_disk_index::<Half>(
            metric,
            &data_path,
            r,
            l,
            &index_path_prefix,
            num_threads,
            search_ram_limit_gb,
            index_build_ram_limit_gb,
            build_pq_bytes as usize,
            use_opq,
        ),
        _ => {
            println!("Unsupported type. Use one of int8, uint8, float or f16.");
            return Err(ANNError::log_index_config_error(
                "data_type".to_string(),
                "Invalid data type".to_string(),
            ));
        }
    };

    match err {
        Ok(_) => {
            println!("Index build completed successfully");
            Ok(())
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            Err(err)
        }
    }
}

fn print_help() {
    println!("Arguments");
    println!("--help, -h                Print information on arguments");
    println!("--data_type               data type <int8/uint8/float> (required)");
    println!("--dist_fn                 distance function <l2/cosine> (required)");
    println!("--data_path               Input data file in bin format (required)");
    println!("--index_path_prefix       Path prefix for saving index file components (required)");
    println!("--max_degree, -R          Maximum graph degree (default: 64)");
    println!("--Lbuild, -L              Build complexity, higher value results in better graphs (default: 100)");
    println!("--search_DRAM_budget      Bound on the memory footprint of the index at search time in GB. Once built, the index will use up only the specified RAM limit, the rest will reside on disk");
    println!("--build_DRAM_budget       Limit on the memory allowed for building the index in GB");
    println!("--num_threads, -T         Number of threads used for building index (defaults to num of CPU logic cores)");
    println!("--build_PQ_bytes          Number of PQ bytes to build the index; 0 for full precision build (default: 0)");
    println!("--use_opq                 Set true for OPQ compression while using PQ distance comparisons for building the index, and false for PQ compression (default: false)");
}
