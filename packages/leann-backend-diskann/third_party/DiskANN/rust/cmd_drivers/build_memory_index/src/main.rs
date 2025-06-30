/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use diskann::{
    common::ANNResult,
    index::create_inmem_index,
    model::{
        vertex::{DIM_104, DIM_128, DIM_256},
        IndexConfiguration, IndexWriteParametersBuilder,
    },
    utils::round_up,
    utils::{load_metadata_from_file, Timer},
};

use vector::{FullPrecisionDistance, Half, Metric};

/// The main function to build an in-memory index
#[allow(clippy::too_many_arguments)]
fn build_in_memory_index<T>(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    alpha: f32,
    save_path: &str,
    num_threads: u32,
    _use_pq_build: bool,
    _num_pq_bytes: usize,
    use_opq: bool,
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
        1f32,
        index_write_parameters,
    );
    let mut index = create_inmem_index::<T>(config)?;

    let timer = Timer::new();

    index.build(data_path, data_num)?;

    let diff = timer.elapsed();

    println!("Indexing time: {}", diff.as_secs_f64());
    index.save(save_path)?;

    Ok(())
}

fn main() -> ANNResult<()> {
    let args = BuildMemoryIndexArgs::parse();

    let _use_pq_build = args.build_pq_bytes > 0;

    println!(
        "Starting index build with R: {}  Lbuild: {}  alpha: {}  #threads: {}",
        args.max_degree, args.l_build, args.alpha, args.num_threads
    );

    let err = match args.data_type {
        DataType::Float => build_in_memory_index::<f32>(
            args.dist_fn,
            &args.data_path.to_string_lossy(),
            args.max_degree,
            args.l_build,
            args.alpha,
            &args.index_path_prefix,
            args.num_threads,
            _use_pq_build,
            args.build_pq_bytes,
            args.use_opq,
        ),
        DataType::FP16 => build_in_memory_index::<Half>(
            args.dist_fn,
            &args.data_path.to_string_lossy(),
            args.max_degree,
            args.l_build,
            args.alpha,
            &args.index_path_prefix,
            args.num_threads,
            _use_pq_build,
            args.build_pq_bytes,
            args.use_opq,
        ),
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum DataType {
    /// Float data type.
    Float,

    /// Half data type.
    FP16,
}

#[derive(Debug, Parser)]
struct BuildMemoryIndexArgs {
    /// data type <int8/uint8/float / fp16> (required)
    #[arg(long = "data_type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist_fn", default_value = "l2")]
    pub dist_fn: Metric,

    /// Path to the data file. The file should be in the format specified by the `data_type` argument.
    #[arg(long = "data_path", short, required = true)]
    pub data_path: PathBuf,

    /// Path to the index file. The index will be saved to this prefixed name.
    #[arg(long = "index_path_prefix", short, required = true)]
    pub index_path_prefix: String,

    /// Number of max out degree from a vertex.
    #[arg(long = "max_degree", short = 'R', default_value = "64")]
    pub max_degree: u32,

    /// Number of candidates to consider when building out edges
    #[arg(long = "l_build", short = 'L', default_value = "100")]
    pub l_build: u32,

    /// alpha controls density and diameter of graph, set 1 for sparse graph, 1.2 or 1.4 for denser graphs with lower diameter
    #[arg(long, short, default_value = "1.2")]
    pub alpha: f32,

    /// Number of threads to use.
    #[arg(long = "num_threads", short = 'T', default_value = "1")]
    pub num_threads: u32,

    /// Number of PQ bytes to build the index; 0 for full precision build
    #[arg(long = "build_pq_bytes", short, default_value = "0")]
    pub build_pq_bytes: usize,

    /// Set true for OPQ compression while using PQ distance comparisons for building the index, and false for PQ compression
    #[arg(long = "use_opq", short, default_value = "false")]
    pub use_opq: bool,
}
