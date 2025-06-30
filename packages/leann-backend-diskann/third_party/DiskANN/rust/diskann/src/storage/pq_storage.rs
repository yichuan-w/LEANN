/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use byteorder::{LittleEndian, ReadBytesExt};
use rand::distributions::{Distribution, Uniform};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;

use crate::common::{ANNError, ANNResult};
use crate::utils::CachedReader;
use crate::utils::{
    convert_types_u32_usize, convert_types_u64_usize, convert_types_usize_u32,
    convert_types_usize_u64, convert_types_usize_u8, save_bin_f32, save_bin_u32, save_bin_u64,
};
use crate::utils::{file_exists, load_bin, open_file_to_write, METADATA_SIZE};

#[derive(Debug)]
pub struct PQStorage {
    /// Pivot table path
    pivot_file: String,

    /// Compressed pivot path
    compressed_pivot_file: String,

    /// Data used to construct PQ table and PQ compressed table
    pq_data_file: String,

    /// PQ data reader
    pq_data_file_reader: File,
}

impl PQStorage {
    pub fn new(
        pivot_file: &str,
        compressed_pivot_file: &str,
        pq_data_file: &str,
    ) -> std::io::Result<Self> {
        let pq_data_file_reader = File::open(pq_data_file)?;
        Ok(Self {
            pivot_file: pivot_file.to_string(),
            compressed_pivot_file: compressed_pivot_file.to_string(),
            pq_data_file: pq_data_file.to_string(),
            pq_data_file_reader,
        })
    }

    pub fn write_compressed_pivot_metadata(&self, npts: i32, pq_chunk: i32) -> std::io::Result<()> {
        let mut writer = open_file_to_write(&self.compressed_pivot_file)?;
        writer.write_all(&npts.to_le_bytes())?;
        writer.write_all(&pq_chunk.to_le_bytes())?;
        Ok(())
    }

    pub fn write_compressed_pivot_data(
        &self,
        compressed_base: &[usize],
        num_centers: usize,
        block_size: usize,
        num_pq_chunks: usize,
    ) -> std::io::Result<()> {
        let mut writer = open_file_to_write(&self.compressed_pivot_file)?;
        writer.seek(SeekFrom::Start((std::mem::size_of::<i32>() * 2) as u64))?;
        if num_centers > 256 {
            writer.write_all(unsafe {
                std::slice::from_raw_parts(
                    compressed_base.as_ptr() as *const u8,
                    block_size * num_pq_chunks * std::mem::size_of::<usize>(),
                )
            })?;
        } else {
            let compressed_base_u8 =
                convert_types_usize_u8(compressed_base, block_size, num_pq_chunks);
            writer.write_all(&compressed_base_u8)?;
        }
        Ok(())
    }

    pub fn write_pivot_data(
        &self,
        full_pivot_data: &[f32],
        centroid: &[f32],
        chunk_offsets: &[usize],
        num_centers: usize,
        dim: usize,
    ) -> std::io::Result<()> {
        let mut cumul_bytes: Vec<usize> = vec![0; 4];
        cumul_bytes[0] = METADATA_SIZE;
        cumul_bytes[1] = cumul_bytes[0]
            + save_bin_f32(
                &self.pivot_file,
                full_pivot_data,
                num_centers,
                dim,
                cumul_bytes[0],
            )?;
        cumul_bytes[2] =
            cumul_bytes[1] + save_bin_f32(&self.pivot_file, centroid, dim, 1, cumul_bytes[1])?;

        // Because the writer only can write u32, u64 but not usize, so we need to convert the type first.
        let chunk_offsets_u64 = convert_types_usize_u32(chunk_offsets, chunk_offsets.len(), 1);
        cumul_bytes[3] = cumul_bytes[2]
            + save_bin_u32(
                &self.pivot_file,
                &chunk_offsets_u64,
                chunk_offsets.len(),
                1,
                cumul_bytes[2],
            )?;

        let cumul_bytes_u64 = convert_types_usize_u64(&cumul_bytes, 4, 1);
        save_bin_u64(&self.pivot_file, &cumul_bytes_u64, cumul_bytes.len(), 1, 0)?;

        Ok(())
    }

    pub fn pivot_data_exist(&self) -> bool {
        file_exists(&self.pivot_file)
    }

    pub fn read_pivot_metadata(&self) -> std::io::Result<(usize, usize)> {
        let (_, file_num_centers, file_dim) = load_bin::<f32>(&self.pivot_file, METADATA_SIZE)?;
        Ok((file_num_centers, file_dim))
    }

    pub fn load_pivot_data(
        &self,
        num_pq_chunks: &usize,
        num_centers: &usize,
        dim: &usize,
    ) -> ANNResult<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        // Load file offset data. File saved as offset data(4*1) -> pivot data(centroid num*dim) -> centroid of dim data(dim*1) -> chunk offset data(chunksize+1*1)
        // Because we only can write u64 rather than usize, so the file stored as u64 type. Need to convert to usize when use.
        let (data, offset_num, nc) = load_bin::<u64>(&self.pivot_file, 0)?;
        let file_offset_data = convert_types_u64_usize(&data, offset_num, nc);
        if offset_num != 4 {
            let error_message = format!("Error reading pq_pivots file {}. Offsets don't contain correct metadata, # offsets = {}, but expecting 4.", &self.pivot_file, offset_num);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, pivot_num, pivot_dim) = load_bin::<f32>(&self.pivot_file, file_offset_data[0])?;
        let full_pivot_data = data;
        if pivot_num != *num_centers || pivot_dim != *dim {
            let error_message = format!("Error reading pq_pivots file {}. file_num_centers = {}, file_dim = {} but expecting {} centers in {} dimensions.", &self.pivot_file, pivot_num, pivot_dim, num_centers, dim);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, centroid_dim, nc) = load_bin::<f32>(&self.pivot_file, file_offset_data[1])?;
        let centroid = data;
        if centroid_dim != *dim || nc != 1 {
            let error_message = format!("Error reading pq_pivots file {}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.", &self.pivot_file, centroid_dim, nc, dim);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, chunk_offset_number, nc) =
            load_bin::<u32>(&self.pivot_file, file_offset_data[2])?;
        let chunk_offsets = convert_types_u32_usize(&data, chunk_offset_number, nc);
        if chunk_offset_number != *num_pq_chunks + 1 || nc != 1 {
            let error_message = format!("Error reading pq_pivots file at chunk offsets; file has nr={}, nc={} but expecting nr={} and nc=1.", chunk_offset_number, nc, num_pq_chunks + 1);
            return Err(ANNError::log_pq_error(error_message));
        }
        Ok((full_pivot_data, centroid, chunk_offsets))
    }

    pub fn read_pq_data_metadata(&mut self) -> std::io::Result<(usize, usize)> {
        let npts_i32 = self.pq_data_file_reader.read_i32::<LittleEndian>()?;
        let dim_i32 = self.pq_data_file_reader.read_i32::<LittleEndian>()?;
        let num_points = npts_i32 as usize;
        let dim = dim_i32 as usize;
        Ok((num_points, dim))
    }

    pub fn read_pq_block_data<T: Copy>(
        &mut self,
        cur_block_size: usize,
        dim: usize,
    ) -> std::io::Result<Vec<T>> {
        let mut buf = vec![0u8; cur_block_size * dim * std::mem::size_of::<T>()];
        self.pq_data_file_reader.read_exact(&mut buf)?;

        let ptr = buf.as_ptr() as *const T;
        let block_data = unsafe { std::slice::from_raw_parts(ptr, cur_block_size * dim) };
        Ok(block_data.to_vec())
    }

    /// streams data from the file, and samples each vector with probability p_val
    /// and returns a matrix of size slice_size* ndims as floating point type.
    /// the slice_size and ndims are set inside the function.
    /// # Arguments
    /// * `file_name` - filename where the data is
    /// * `p_val` - possibility to sample data
    /// * `sampled_vectors` - sampled vector chose by p_val possibility
    /// * `slice_size` - how many sampled data return
    /// * `dim` - each sample data dimension
    pub fn gen_random_slice<T: Default + Copy + Into<f32>>(
        &self,
        mut p_val: f64,
    ) -> ANNResult<(Vec<f32>, usize, usize)> {
        let read_blk_size = 64 * 1024 * 1024;
        let mut reader = CachedReader::new(&self.pq_data_file, read_blk_size)?;

        let npts = reader.read_u32()? as usize;
        let dim = reader.read_u32()? as usize;
        let mut sampled_vectors: Vec<f32> = Vec::new();
        let mut slice_size = 0;
        p_val = if p_val < 1f64 { p_val } else { 1f64 };

        let mut generator = rand::thread_rng();
        let distribution = Uniform::from(0.0..1.0);

        for _ in 0..npts {
            let mut cur_vector_bytes = vec![0u8; dim * mem::size_of::<T>()];
            reader.read(&mut cur_vector_bytes)?;
            let random_value = distribution.sample(&mut generator);
            if random_value < p_val {
                let ptr = cur_vector_bytes.as_ptr() as *const T;
                let cur_vector_t = unsafe { std::slice::from_raw_parts(ptr, dim) };
                sampled_vectors.extend(cur_vector_t.iter().map(|&t| t.into()));
                slice_size += 1;
            }
        }

        Ok((sampled_vectors, slice_size, dim))
    }
}

#[cfg(test)]
mod pq_storage_tests {
    use rand::Rng;

    use super::*;
    use crate::utils::gen_random_slice;

    const DATA_FILE: &str = "tests/data/siftsmall_learn.bin";
    const PQ_PIVOT_PATH: &str = "tests/data/siftsmall_learn.bin_pq_pivots.bin";
    const PQ_COMPRESSED_PATH: &str = "tests/data/empty_pq_compressed.bin";

    #[test]
    fn new_test() {
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, DATA_FILE);
        assert!(result.is_ok());
    }

    #[test]
    fn write_compressed_pivot_metadata_test() {
        let compress_pivot_path = "write_compressed_pivot_metadata_test.bin";
        let result = PQStorage::new(PQ_PIVOT_PATH, compress_pivot_path, DATA_FILE).unwrap();

        _ = result.write_compressed_pivot_metadata(100, 20);
        let mut result_reader = File::open(compress_pivot_path).unwrap();
        let npts_i32 = result_reader.read_i32::<LittleEndian>().unwrap();
        let dim_i32 = result_reader.read_i32::<LittleEndian>().unwrap();

        assert_eq!(npts_i32, 100);
        assert_eq!(dim_i32, 20);

        std::fs::remove_file(compress_pivot_path).unwrap();
    }

    #[test]
    fn write_compressed_pivot_data_test() {
        let compress_pivot_path = "write_compressed_pivot_data_test.bin";
        let result = PQStorage::new(PQ_PIVOT_PATH, compress_pivot_path, DATA_FILE).unwrap();

        let mut rng = rand::thread_rng();

        let num_centers = 256;
        let block_size = 4;
        let num_pq_chunks = 2;
        let compressed_base: Vec<usize> = (0..block_size * num_pq_chunks)
            .map(|_| rng.gen_range(0..num_centers))
            .collect();
        _ = result.write_compressed_pivot_data(
            &compressed_base,
            num_centers,
            block_size,
            num_pq_chunks,
        );

        let mut result_reader = File::open(compress_pivot_path).unwrap();
        _ = result_reader.read_i32::<LittleEndian>().unwrap();
        _ = result_reader.read_i32::<LittleEndian>().unwrap();
        let mut buf = vec![0u8; block_size * num_pq_chunks * std::mem::size_of::<u8>()];
        result_reader.read_exact(&mut buf).unwrap();

        let ptr = buf.as_ptr() as *const u8;
        let block_data = unsafe { std::slice::from_raw_parts(ptr, block_size * num_pq_chunks) };

        for index in 0..block_data.len() {
            assert_eq!(compressed_base[index], block_data[index] as usize);
        }
        std::fs::remove_file(compress_pivot_path).unwrap();
    }

    #[test]
    fn pivot_data_exist_test() {
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, DATA_FILE).unwrap();
        assert!(result.pivot_data_exist());

        let pivot_path = "not_exist_pivot_path.bin";
        let result = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, DATA_FILE).unwrap();
        assert!(!result.pivot_data_exist());
    }

    #[test]
    fn read_pivot_metadata_test() {
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, DATA_FILE).unwrap();
        let (npt, dim) = result.read_pivot_metadata().unwrap();

        assert_eq!(npt, 256);
        assert_eq!(dim, 128);
    }

    #[test]
    fn load_pivot_data_test() {
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, DATA_FILE).unwrap();
        let (pq_pivot_data, centroids, chunk_offsets) =
            result.load_pivot_data(&1, &256, &128).unwrap();

        assert_eq!(pq_pivot_data.len(), 256 * 128);
        assert_eq!(centroids.len(), 128);
        assert_eq!(chunk_offsets.len(), 2);
    }

    #[test]
    fn read_pq_data_metadata_test() {
        let mut result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, DATA_FILE).unwrap();
        let (npt, dim) = result.read_pq_data_metadata().unwrap();

        assert_eq!(npt, 25000);
        assert_eq!(dim, 128);
    }

    #[test]
    fn gen_random_slice_test() {
        let file_name = "gen_random_slice_test.bin";
        //npoints=2, dim=8
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let (sampled_vectors, slice_size, ndims) =
            gen_random_slice::<f32>(file_name, 1f64).unwrap();
        let mut start = 8;
        (0..sampled_vectors.len()).for_each(|i| {
            assert_eq!(sampled_vectors[i].to_le_bytes(), data[start..start + 4]);
            start += 4;
        });
        assert_eq!(sampled_vectors.len(), 16);
        assert_eq!(slice_size, 2);
        assert_eq!(ndims, 8);

        let (sampled_vectors, slice_size, ndims) =
            gen_random_slice::<f32>(file_name, 0f64).unwrap();
        assert_eq!(sampled_vectors.len(), 0);
        assert_eq!(slice_size, 0);
        assert_eq!(ndims, 8);

        std::fs::remove_file(file_name).expect("Failed to delete file");
    }
}
