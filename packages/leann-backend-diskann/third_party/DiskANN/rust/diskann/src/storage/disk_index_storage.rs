/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::{fs, mem};

use crate::common::{ANNError, ANNResult};
use crate::model::NUM_PQ_CENTROIDS;
use crate::storage::PQStorage;
use crate::utils::{convert_types_u32_usize, convert_types_u64_usize, load_bin, save_bin_u64};
use crate::utils::{
    file_exists, gen_sample_data, get_file_size, round_up, CachedReader, CachedWriter,
};

const SECTOR_LEN: usize = 4096;

/// Todo: Remove the allow(dead_code) when the disk search code is complete
#[allow(dead_code)]
pub struct PQPivotData {
    dim: usize,
    pq_table: Vec<f32>,
    centroids: Vec<f32>,
    chunk_offsets: Vec<usize>,
}

pub struct DiskIndexStorage<T> {
    /// Dataset file
    dataset_file: String,

    /// Index file path prefix
    index_path_prefix: String,

    // TODO: Only a placeholder for T, will be removed later
    _marker: PhantomData<T>,

    pq_storage: PQStorage,
}

impl<T> DiskIndexStorage<T> {
    /// Create DiskIndexStorage instance
    pub fn new(dataset_file: String, index_path_prefix: String) -> ANNResult<Self> {
        let pq_storage: PQStorage = PQStorage::new(
            &(index_path_prefix.clone() + ".bin_pq_pivots.bin"),
            &(index_path_prefix.clone() + ".bin_pq_compressed.bin"),
            &dataset_file,
        )?;

        Ok(DiskIndexStorage {
            dataset_file,
            index_path_prefix,
            _marker: PhantomData,
            pq_storage,
        })
    }

    pub fn get_pq_storage(&mut self) -> &mut PQStorage {
        &mut self.pq_storage
    }

    pub fn dataset_file(&self) -> &String {
        &self.dataset_file
    }

    pub fn index_path_prefix(&self) -> &String {
        &self.index_path_prefix
    }

    /// Create disk layout
    /// Sector #1: disk_layout_meta
    /// Sector #n: num_nodes_per_sector nodes
    /// Each node's layout: {full precision vector:[T; DIM]}{num_nbrs: u32}{neighbors: [u32; num_nbrs]}
    /// # Arguments
    /// * `dataset_file` - dataset file containing full precision vectors
    /// * `mem_index_file` - in-memory index graph file
    /// * `disk_layout_file` - output disk layout file
    pub fn create_disk_layout(&self) -> ANNResult<()> {
        let mem_index_file = self.mem_index_file();
        let disk_layout_file = self.disk_index_file();

        // amount to read or write in one shot
        let read_blk_size = 64 * 1024 * 1024;
        let write_blk_size = read_blk_size;
        let mut dataset_reader = CachedReader::new(self.dataset_file.as_str(), read_blk_size)?;

        let num_pts = dataset_reader.read_u32()? as u64;
        let dims = dataset_reader.read_u32()? as u64;

        // Create cached reader + writer
        let actual_file_size = get_file_size(mem_index_file.as_str())?;
        println!("Vamana index file size={}", actual_file_size);

        let mut vamana_reader = File::open(mem_index_file)?;
        let mut diskann_writer = CachedWriter::new(disk_layout_file.as_str(), write_blk_size)?;

        let index_file_size = vamana_reader.read_u64::<LittleEndian>()?;
        if index_file_size != actual_file_size {
            println!(
                "Vamana Index file size does not match expected size per meta-data. file size from file: {}, actual file size: {}",
                index_file_size, actual_file_size
            );
        }

        let max_degree = vamana_reader.read_u32::<LittleEndian>()?;
        let medoid = vamana_reader.read_u32::<LittleEndian>()?;
        let vamana_frozen_num = vamana_reader.read_u64::<LittleEndian>()?;

        let mut vamana_frozen_loc = 0;
        if vamana_frozen_num == 1 {
            vamana_frozen_loc = medoid;
        }

        let max_node_len = ((max_degree as u64 + 1) * (mem::size_of::<u32>() as u64))
            + (dims * (mem::size_of::<T>() as u64));
        let num_nodes_per_sector = (SECTOR_LEN as u64) / max_node_len;

        println!("medoid: {}B", medoid);
        println!("max_node_len: {}B", max_node_len);
        println!("num_nodes_per_sector: {}B", num_nodes_per_sector);

        // SECTOR_LEN buffer for each sector
        let mut sector_buf = vec![0u8; SECTOR_LEN];
        let mut node_buf = vec![0u8; max_node_len as usize];

        let num_nbrs_start = (dims as usize) * mem::size_of::<T>();
        let nbrs_buf_start = num_nbrs_start + mem::size_of::<u32>();

        // number of sectors (1 for meta data)
        let num_sectors = round_up(num_pts, num_nodes_per_sector) / num_nodes_per_sector;
        let disk_index_file_size = (num_sectors + 1) * (SECTOR_LEN as u64);

        let disk_layout_meta = vec![
            num_pts,
            dims,
            medoid as u64,
            max_node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc as u64,
            // append_reorder_data
            // We are not supporting this. Temporarily write it into the layout so that
            // we can leverage C++ query driver to test the disk index
            false as u64,
            disk_index_file_size,
        ];

        diskann_writer.write(&sector_buf)?;

        let mut cur_node_coords = vec![0u8; (dims as usize) * mem::size_of::<T>()];
        let mut cur_node_id = 0u64;

        for sector in 0..num_sectors {
            if sector % 100_000 == 0 {
                println!("Sector #{} written", sector);
            }
            sector_buf.fill(0);

            for sector_node_id in 0..num_nodes_per_sector {
                if cur_node_id >= num_pts {
                    break;
                }

                node_buf.fill(0);

                // read cur node's num_nbrs
                let num_nbrs = vamana_reader.read_u32::<LittleEndian>()?;

                // sanity checks on num_nbrs
                debug_assert!(num_nbrs > 0);
                debug_assert!(num_nbrs <= max_degree);

                // write coords of node first
                dataset_reader.read(&mut cur_node_coords)?;
                node_buf[..cur_node_coords.len()].copy_from_slice(&cur_node_coords);

                // write num_nbrs
                LittleEndian::write_u32(
                    &mut node_buf[num_nbrs_start..(num_nbrs_start + mem::size_of::<u32>())],
                    num_nbrs,
                );

                // write neighbors
                let nbrs_buf = &mut node_buf[nbrs_buf_start
                    ..(nbrs_buf_start + (num_nbrs as usize) * mem::size_of::<u32>())];
                vamana_reader.read_exact(nbrs_buf)?;

                // get offset into sector_buf
                let sector_node_buf_start = (sector_node_id * max_node_len) as usize;
                let sector_node_buf = &mut sector_buf
                    [sector_node_buf_start..(sector_node_buf_start + max_node_len as usize)];
                sector_node_buf.copy_from_slice(&node_buf[..(max_node_len as usize)]);

                cur_node_id += 1;
            }

            // flush sector to disk
            diskann_writer.write(&sector_buf)?;
        }

        diskann_writer.flush()?;
        save_bin_u64(
            disk_layout_file.as_str(),
            &disk_layout_meta,
            disk_layout_meta.len(),
            1,
            0,
        )?;

        Ok(())
    }

    pub fn index_build_cleanup(&self) -> ANNResult<()> {
        fs::remove_file(self.mem_index_file())?;
        Ok(())
    }

    pub fn gen_query_warmup_data(&self, sampling_rate: f64) -> ANNResult<()> {
        gen_sample_data::<T>(
            &self.dataset_file,
            &self.warmup_query_prefix(),
            sampling_rate,
        )?;
        Ok(())
    }

    /// Load pre-trained pivot table
    pub fn load_pq_pivots_bin(
        &self,
        num_pq_chunks: &usize,
    ) -> ANNResult<PQPivotData> {
        let pq_pivots_path = &self.pq_pivot_file();
        if !file_exists(pq_pivots_path) {
            return Err(ANNError::log_pq_error(
                "ERROR: PQ k-means pivot file not found.".to_string(),
            ));
        }

        let (data, offset_num, offset_dim) = load_bin::<u64>(pq_pivots_path, 0)?;
        let file_offset_data = convert_types_u64_usize(&data, offset_num, offset_dim);
        if offset_num != 4 {
            let error_message = format!("Error reading pq_pivots file {}. Offsets don't contain correct metadata, # offsets = {}, but expecting 4.", pq_pivots_path, offset_num);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, pivot_num, dim) = load_bin::<f32>(pq_pivots_path, file_offset_data[0])?;
        let pq_table = data.to_vec();
        if pivot_num != NUM_PQ_CENTROIDS {
            let error_message = format!(
                "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                pq_pivots_path, pivot_num, NUM_PQ_CENTROIDS
            );
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, centroid_dim, nc) = load_bin::<f32>(pq_pivots_path, file_offset_data[1])?;
        let centroids = data.to_vec();
        if centroid_dim != dim || nc != 1 {
            let error_message = format!("Error reading pq_pivots file {}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.", pq_pivots_path, centroid_dim, nc, dim);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, chunk_offset_num, nc) = load_bin::<u32>(pq_pivots_path, file_offset_data[2])?;
        let chunk_offsets = convert_types_u32_usize(&data, chunk_offset_num, nc);
        if chunk_offset_num != num_pq_chunks + 1 || nc != 1 {
            let error_message = format!("Error reading pq_pivots file at chunk offsets; file has nr={}, nc={} but expecting nr={} and nc=1.", chunk_offset_num, nc, num_pq_chunks + 1);
            return Err(ANNError::log_pq_error(error_message));
        }

        Ok(PQPivotData {
            dim, 
            pq_table, 
            centroids, 
            chunk_offsets
        })
    }

    fn mem_index_file(&self) -> String {
        self.index_path_prefix.clone() + "_mem.index"
    }

    fn disk_index_file(&self) -> String {
        self.index_path_prefix.clone() + "_disk.index"
    }

    fn warmup_query_prefix(&self) -> String {
        self.index_path_prefix.clone() + "_sample"
    }

    pub fn pq_pivot_file(&self) -> String {
        self.index_path_prefix.clone() + ".bin_pq_pivots.bin"
    }

    pub fn compressed_pq_pivot_file(&self) -> String {
        self.index_path_prefix.clone() + ".bin_pq_compressed.bin"
    }
}

#[cfg(test)]
mod disk_index_storage_test {
    use std::fs;

    use crate::test_utils::get_test_file_path;

    use super::*;

    const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
    const DISK_INDEX_PATH_PREFIX: &str = "tests/data/disk_index_siftsmall_learn_256pts_R4_L50_A1.2";
    const TRUTH_DISK_LAYOUT: &str =
        "tests/data/truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_disk.index";

    #[test]
    fn create_disk_layout_test() {
        let storage = DiskIndexStorage::<f32>::new(
            get_test_file_path(TEST_DATA_FILE),
            get_test_file_path(DISK_INDEX_PATH_PREFIX),
        ).unwrap();
        storage.create_disk_layout().unwrap();

        let disk_layout_file = storage.disk_index_file();
        let rust_disk_layout = fs::read(disk_layout_file.as_str()).unwrap();
        let truth_disk_layout = fs::read(get_test_file_path(TRUTH_DISK_LAYOUT).as_str()).unwrap();

        assert!(rust_disk_layout == truth_disk_layout);

        fs::remove_file(disk_layout_file.as_str()).expect("Failed to delete file");
    }

    #[test]
    fn load_pivot_test() {
        let dim: usize = 128;
        let num_pq_chunk: usize = 1;
        let pivot_file_prefix: &str = "tests/data/siftsmall_learn";
        let storage = DiskIndexStorage::<f32>::new(
            get_test_file_path(TEST_DATA_FILE),
            pivot_file_prefix.to_string(),
        ).unwrap();            

        let pq_pivot_data =
            storage.load_pq_pivots_bin(&num_pq_chunk).unwrap();

        assert_eq!(pq_pivot_data.pq_table.len(), NUM_PQ_CENTROIDS * dim);
        assert_eq!(pq_pivot_data.centroids.len(), dim);

        assert_eq!(pq_pivot_data.chunk_offsets[0], 0);
        assert_eq!(pq_pivot_data.chunk_offsets[1], dim);
        assert_eq!(pq_pivot_data.chunk_offsets.len(), num_pq_chunk + 1);
    }

    #[test]
    #[should_panic(expected = "ERROR: PQ k-means pivot file not found.")]
    fn load_pivot_file_not_exist_test() {
        let num_pq_chunk: usize = 1;
        let pivot_file_prefix: &str = "tests/data/siftsmall_learn_file_not_exist";
        let storage = DiskIndexStorage::<f32>::new(
            get_test_file_path(TEST_DATA_FILE),
            pivot_file_prefix.to_string(),
        ).unwrap();
        let _ = storage.load_pq_pivots_bin(&num_pq_chunk).unwrap();
    }
}
