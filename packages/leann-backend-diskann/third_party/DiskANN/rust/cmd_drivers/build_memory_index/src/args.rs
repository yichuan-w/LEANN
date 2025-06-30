use clap::{Args, Parser};

#[derive(Debug, Args)]
enum DataType {
    /// Float data type.
    Float,

    /// Half data type.
    FP16,
}

#[derive(Debug, Args)]
enum DistanceFunction {
    /// Euclidean distance.
    L2,

    /// Cosine distance.
    Cosine,
}

#[derive(Debug, Parser)]
struct BuildMemoryIndexArgs {
    /// Data type of the vectors.
    #[clap(long, default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[clap(long, default_value = "l2")]
    pub dist_fn: Metric,

    /// Path to the data file. The file should be in the format specified by the `data_type` argument.
    #[clap(long, short, required = true)]
    pub data_path: String,

    /// Path to the index file. The index will be saved to this prefixed name.
    #[clap(long, short, required = true)]
    pub index_path_prefix: String,

    /// Number of max out degree from a vertex.
    #[clap(long, default_value = "32")]
    pub max_degree: usize,

    /// Number of candidates to consider when building out edges
    #[clap(long, short default_value = "50")]
    pub l_build: usize,

    /// Alpha to use to build diverse edges
    #[clap(long, short default_value = "1.0")]
    pub alpha: f32,

    /// Number of threads to use.
    #[clap(long, short, default_value = "1")]
    pub num_threads: u8,

    /// Number of PQ bytes to use.
    #[clap(long, short, default_value = "8")]
    pub build_pq_bytes: usize,

    /// Use opq?
    #[clap(long, short, default_value = "false")]
    pub use_opq: bool,
}
