/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::mem;

use logger::logger::indexlog::DiskIndexConstructionCheckpoint;
use vector::FullPrecisionDistance;

use crate::common::{ANNResult, ANNError};
use crate::index::{InmemIndex, ANNInmemIndex};
use crate::instrumentation::DiskIndexBuildLogger;
use crate::model::configuration::DiskIndexBuildParameters;
use crate::model::{IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE, MAX_PQ_CHUNKS, generate_quantized_data, GRAPH_SLACK_FACTOR};
use crate::storage::DiskIndexStorage;
use crate::utils::set_rayon_num_threads;

use super::ann_disk_index::ANNDiskIndex;

pub const OVERHEAD_FACTOR: f64 = 1.1f64;

pub const MAX_SAMPLE_POINTS_FOR_WARMUP: usize = 100_000;

pub struct DiskIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Parameters for index construction
    /// None for query path
    disk_build_param: Option<DiskIndexBuildParameters>,

    configuration: IndexConfiguration, 

    pub storage: DiskIndexStorage<T>,
}

impl<T, const N: usize> DiskIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn new(
        disk_build_param: Option<DiskIndexBuildParameters>, 
        configuration: IndexConfiguration, 
        storage: DiskIndexStorage<T>,
    ) -> Self {
        Self {
            disk_build_param,
            configuration,
            storage,
        }
    }

    pub fn disk_build_param(&self) -> &Option<DiskIndexBuildParameters> {
        &self.disk_build_param
    }

    pub fn index_configuration(&self) -> &IndexConfiguration {
        &self.configuration
    }

    fn build_inmem_index(&self, num_points: usize, data_path: &str, inmem_index_path: &str) -> ANNResult<()> {
        let estimated_index_ram = self.estimate_ram_usage(num_points);
        if estimated_index_ram >= self.fetch_disk_build_param()?.index_build_ram_limit() * 1024_f64 * 1024_f64 * 1024_f64 {
            return Err(ANNError::log_index_error(format!(
                "Insufficient memory budget for index build, index_build_ram_limit={}GB estimated_index_ram={}GB",
                self.fetch_disk_build_param()?.index_build_ram_limit(),
                estimated_index_ram / (1024_f64 * 1024_f64 * 1024_f64),
            )));
        }

        let mut index = InmemIndex::<T, N>::new(self.configuration.clone())?;
        index.build(data_path, num_points)?;
        index.save(inmem_index_path)?;

        Ok(())
    }

    #[inline]
    fn estimate_ram_usage(&self, size: usize) -> f64 {
        let degree = self.configuration.index_write_parameter.max_degree as usize;
        let datasize = mem::size_of::<T>();

        let dataset_size = (size * N * datasize) as f64;
        let graph_size = (size * degree * mem::size_of::<u32>()) as f64 * GRAPH_SLACK_FACTOR;
        
        OVERHEAD_FACTOR * (dataset_size + graph_size)
    }

    #[inline]
    fn fetch_disk_build_param(&self) -> ANNResult<&DiskIndexBuildParameters> {
        self.disk_build_param
            .as_ref()
            .ok_or_else(|| ANNError::log_index_config_error(
                "disk_build_param".to_string(), 
                "disk_build_param is None".to_string()))
    }
}

impl<T, const N: usize> ANNDiskIndex<T> for DiskIndex<T, N> 
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn build(&mut self, codebook_prefix: &str) -> ANNResult<()> {
        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        println!("Starting index build: R={} L={} Query RAM budget={} Indexing RAM budget={} T={}",
            self.configuration.index_write_parameter.max_degree, 
            self.configuration.index_write_parameter.search_list_size,
            self.fetch_disk_build_param()?.search_ram_limit(),
            self.fetch_disk_build_param()?.index_build_ram_limit(),
            self.configuration.index_write_parameter.num_threads
        );

        let mut logger = DiskIndexBuildLogger::new(DiskIndexConstructionCheckpoint::PqConstruction);

        // PQ memory consumption = PQ pivots + PQ compressed table
        // PQ pivots: dim * num_centroids * sizeof::<T>()
        // PQ compressed table: num_pts * num_pq_chunks * (dim / num_pq_chunks) * sizeof::<u8>()
        // * Because num_centroids is 256, centroid id can be represented by u8
        let num_points = self.configuration.max_points;
        let dim = self.configuration.dim;
        let p_val = MAX_PQ_TRAINING_SET_SIZE / (num_points as f64);
        let mut num_pq_chunks = ((self.fetch_disk_build_param()?.search_ram_limit() / (num_points as f64)).floor()) as usize;
        num_pq_chunks = if num_pq_chunks == 0 { 1 } else { num_pq_chunks };
        num_pq_chunks = if num_pq_chunks > dim { dim } else { num_pq_chunks };
        num_pq_chunks = if num_pq_chunks > MAX_PQ_CHUNKS { MAX_PQ_CHUNKS } else { num_pq_chunks };

        println!("Compressing {}-dimensional data into {} bytes per vector.", dim, num_pq_chunks);

        // TODO: Decouple PQ from file access
        generate_quantized_data::<T>(
            p_val,
            num_pq_chunks,
            codebook_prefix,
            self.storage.get_pq_storage(),
        )?;
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::InmemIndexBuild)?;

        // TODO: Decouple index from file access
        let inmem_index_path = self.storage.index_path_prefix().clone() + "_mem.index";
        self.build_inmem_index(num_points, self.storage.dataset_file(), inmem_index_path.as_str())?;
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::DiskLayout)?;

        self.storage.create_disk_layout()?;
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::None)?;

        let ten_percent_points = ((num_points as f64) * 0.1_f64).ceil();
        let num_sample_points = if ten_percent_points > (MAX_SAMPLE_POINTS_FOR_WARMUP as f64) { MAX_SAMPLE_POINTS_FOR_WARMUP as f64 } else { ten_percent_points };
        let sample_sampling_rate = num_sample_points / (num_points as f64);
        self.storage.gen_query_warmup_data(sample_sampling_rate)?;

        self.storage.index_build_cleanup()?;

        Ok(())
    }
}

