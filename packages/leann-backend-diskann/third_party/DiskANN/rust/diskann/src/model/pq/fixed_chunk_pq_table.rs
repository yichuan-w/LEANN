/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations)]

use hashbrown::HashMap;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSliceMut,
};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

use crate::{
    common::{ANNError, ANNResult},
    model::NUM_PQ_CENTROIDS,
};

/// PQ Pivot table loading and calculate distance
#[derive(Debug)]
pub struct FixedChunkPQTable {
    /// pq_tables = float array of size [256 * ndims]
    pq_table: Vec<f32>,

    /// ndims = true dimension of vectors
    dim: usize,

    /// num_pq_chunks = the pq chunk number
    num_pq_chunks: usize,

    /// chunk_offsets = the offset of each chunk, start from 0
    chunk_offsets: Vec<usize>,

    /// centroid of each dimension
    centroids: Vec<f32>,

    /// Becasue we're using L2 distance, this is no needed now.
    /// Transport of pq_table. transport_pq_table = float array of size [ndims * 256].
    /// e.g. if pa_table is 2 centroids * 3 dims
    /// [ 1, 2, 3,
    ///   4, 5, 6]
    /// then transport_pq_table would be 3 dims * 2 centroids
    /// [ 1, 4,
    ///   2, 5,
    ///   3, 6]
    /// transport_pq_table: Vec<f32>,

    /// Map dim offset to chunk index e.g., 8 dims in to 2 chunks
    /// then would be [(0,0), (1,0), (2,0), (3,0), (4,1), (5,1), (6,1), (7,1)]
    dimoffset_chunk_mapping: HashMap<usize, usize>,
}

impl FixedChunkPQTable {
    /// Create the FixedChunkPQTable with dim and chunk numbers and pivot file data (pivot table + cenroids + chunk offsets)
    pub fn new(
        dim: usize,
        num_pq_chunks: usize,
        pq_table: Vec<f32>,
        centroids: Vec<f32>,
        chunk_offsets: Vec<usize>,
    ) -> Self {
        let mut dimoffset_chunk_mapping = HashMap::new();
        for chunk_index in 0..num_pq_chunks {
            for dim_offset in chunk_offsets[chunk_index]..chunk_offsets[chunk_index + 1] {
                dimoffset_chunk_mapping.insert(dim_offset, chunk_index);
            }
        }

        Self {
            pq_table,
            dim,
            num_pq_chunks,
            chunk_offsets,
            centroids,
            dimoffset_chunk_mapping,
        }
    }

    /// Get chunk number
    pub fn get_num_chunks(&self) -> usize {
        self.num_pq_chunks
    }

    /// Shifting the query according to mean or the whole corpus
    pub fn preprocess_query(&self, query_vec: &mut [f32]) {
        for (query, &centroid) in query_vec.iter_mut().zip(self.centroids.iter()) {
            *query -= centroid;
        }
    }

    /// Pre-calculated the distance between query and each centroid by l2 distance
    /// * `query_vec` - query vector: 1 * dim
    /// * `dist_vec` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
    #[allow(clippy::needless_range_loop)]
    pub fn populate_chunk_distances(&self, query_vec: &[f32]) -> Vec<f32> {
        let mut dist_vec = vec![0.0; self.num_pq_chunks * NUM_PQ_CENTROIDS];
        for centroid_index in 0..NUM_PQ_CENTROIDS {
            for chunk_index in 0..self.num_pq_chunks {
                for dim_offset in
                    self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1]
                {
                    let diff: f32 = self.pq_table[self.dim * centroid_index + dim_offset]
                        - query_vec[dim_offset];
                    dist_vec[chunk_index * NUM_PQ_CENTROIDS + centroid_index] += diff * diff;
                }
            }
        }
        dist_vec
    }

    /// Pre-calculated the distance between query and each centroid by inner product
    /// * `query_vec` - query vector: 1 * dim
    /// * `dist_vec` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
    ///
    /// Reason to allow clippy::needless_range_loop:
    /// The inner loop is operating over a range that is different for each iteration of the outer loop.
    /// This isn't a scenario where using iter().enumerate() would be easily applicable,
    /// because the inner loop isn't iterating directly over the contents of a slice or array.
    /// Thus, using indexing might be the most straightforward way to express this logic.
    #[allow(clippy::needless_range_loop)]
    pub fn populate_chunk_inner_products(&self, query_vec: &[f32]) -> Vec<f32> {
        let mut dist_vec = vec![0.0; self.num_pq_chunks * NUM_PQ_CENTROIDS];
        for centroid_index in 0..NUM_PQ_CENTROIDS {
            for chunk_index in 0..self.num_pq_chunks {
                for dim_offset in
                    self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1]
                {
                    // assumes that we are not shifting the vectors to mean zero, i.e., centroid
                    // array should be all zeros returning negative to keep the search code
                    // clean (max inner product vs min distance)
                    let diff: f32 = self.pq_table[self.dim * centroid_index + dim_offset]
                        * query_vec[dim_offset];
                    dist_vec[chunk_index * NUM_PQ_CENTROIDS + centroid_index] -= diff;
                }
            }
        }
        dist_vec
    }

    /// Calculate the distance between query and given centroid by l2 distance
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    #[allow(clippy::needless_range_loop)]
    pub fn l2_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        let mut res_vec: Vec<f32> = vec![0.0; self.num_pq_chunks];
        res_vec
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk_index, chunk_diff)| {
                for dim_offset in
                    self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1]
                {
                    let diff = self.pq_table
                        [self.dim * base_vec[chunk_index] as usize + dim_offset]
                        - query_vec[dim_offset];
                    *chunk_diff += diff * diff;
                }
            });

        let res: f32 = res_vec.iter().sum::<f32>();

        res
    }

    /// Calculate the distance between query and given centroid by inner product
    /// * `query_vec` - query vector: 1 * dim
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    #[allow(clippy::needless_range_loop)]
    pub fn inner_product(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        let mut res_vec: Vec<f32> = vec![0.0; self.num_pq_chunks];
        res_vec
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk_index, chunk_diff)| {
                for dim_offset in
                    self.chunk_offsets[chunk_index]..self.chunk_offsets[chunk_index + 1]
                {
                    *chunk_diff += self.pq_table
                        [self.dim * base_vec[chunk_index] as usize + dim_offset]
                        * query_vec[dim_offset];
                }
            });

        let res: f32 = res_vec.iter().sum::<f32>();

        // returns negative value to simulate distances (max -> min conversion)
        -res
    }

    /// Revert vector by adding centroid
    /// * `base_vec` - given centroid array: 1 * num_pq_chunks
    /// * `out_vec` - reverted vector
    pub fn inflate_vector(&self, base_vec: &[u8]) -> ANNResult<Vec<f32>> {
        let mut out_vec: Vec<f32> = vec![0.0; self.dim];
        for (dim_offset, value) in out_vec.iter_mut().enumerate() {
            let chunk_index =
                self.dimoffset_chunk_mapping
                    .get(&dim_offset)
                    .ok_or(ANNError::log_pq_error(
                        "ERROR: dim_offset not found in dimoffset_chunk_mapping".to_string(),
                    ))?;
            *value = self.pq_table[self.dim * base_vec[*chunk_index] as usize + dim_offset]
                + self.centroids[dim_offset];
        }

        Ok(out_vec)
    }
}

/// Given a batch input nodes, return a batch of PQ distance
/// * `pq_ids` - batch nodes: n_pts * pq_nchunks
/// * `n_pts` - batch number
/// * `pq_nchunks` - pq chunk number number
/// * `pq_dists` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
/// * `dists_out` - n_pts * 1
pub fn pq_dist_lookup(
    pq_ids: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
) -> Vec<f32> {
    let mut dists_out: Vec<f32> = vec![0.0; n_pts];
    unsafe {
        _mm_prefetch(dists_out.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(64) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(128) as *const i8, _MM_HINT_T0);
    }
    for chunk in 0..pq_nchunks {
        let chunk_dists = &pq_dists[256 * chunk..];
        if chunk < pq_nchunks - 1 {
            unsafe {
                _mm_prefetch(
                    chunk_dists.as_ptr().offset(256 * chunk as isize).add(256) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        dists_out
            .par_iter_mut()
            .enumerate()
            .for_each(|(n_iter, dist)| {
                let pq_centerid = pq_ids[pq_nchunks * n_iter + chunk];
                *dist += chunk_dists[pq_centerid as usize];
            });
    }
    dists_out
}

pub fn aggregate_coords(ids: &[u32], all_coords: &[u8], ndims: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; ids.len() * ndims];
    let ndim_u32 = ndims as u32;
    out.par_chunks_mut(ndims)
        .enumerate()
        .for_each(|(index, chunk)| {
            let id_compressed_pivot = &all_coords
                [(ids[index] * ndim_u32) as usize..(ids[index] * ndim_u32 + ndim_u32) as usize];
            let temp_slice =
                unsafe { std::slice::from_raw_parts(id_compressed_pivot.as_ptr(), ndims) };
            chunk.copy_from_slice(temp_slice);
        });

    out
}

#[cfg(test)]
mod fixed_chunk_pq_table_test {

    use super::*;
    use crate::common::{ANNError, ANNResult};
    use crate::utils::{convert_types_u32_usize, convert_types_u64_usize, file_exists, load_bin};

    const DIM: usize = 128;

    #[test]
    fn load_pivot_test() {
        let pq_pivots_path: &str = "tests/data/siftsmall_learn.bin_pq_pivots.bin";
        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1).unwrap();
        let fixed_chunk_pq_table =
            FixedChunkPQTable::new(dim, 1, pq_table, centroids, chunk_offsets);

        assert_eq!(dim, DIM);
        assert_eq!(fixed_chunk_pq_table.pq_table.len(), DIM * NUM_PQ_CENTROIDS);
        assert_eq!(fixed_chunk_pq_table.centroids.len(), DIM);

        assert_eq!(fixed_chunk_pq_table.chunk_offsets[0], 0);
        assert_eq!(fixed_chunk_pq_table.chunk_offsets[1], DIM);
        assert_eq!(fixed_chunk_pq_table.chunk_offsets.len(), 2);
    }

    #[test]
    fn get_num_chunks_test() {
        let num_chunks = 7;
        let pa_table = vec![0.0; DIM * NUM_PQ_CENTROIDS];
        let centroids = vec![0.0; DIM];
        let chunk_offsets = vec![0, 7, 9, 11, 22, 34, 78, 127];
        let fixed_chunk_pq_table =
            FixedChunkPQTable::new(DIM, num_chunks, pa_table, centroids, chunk_offsets);
        let chunk: usize = fixed_chunk_pq_table.get_num_chunks();
        assert_eq!(chunk, num_chunks);
    }

    #[test]
    fn preprocess_query_test() {
        let pq_pivots_path: &str = "tests/data/siftsmall_learn.bin_pq_pivots.bin";
        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1).unwrap();
        let fixed_chunk_pq_table =
            FixedChunkPQTable::new(dim, 1, pq_table, centroids, chunk_offsets);

        let mut query_vec: Vec<f32> = vec![
            32.39f32, 78.57f32, 50.32f32, 80.46f32, 6.47f32, 69.76f32, 94.2f32, 83.36f32, 5.8f32,
            68.78f32, 42.32f32, 61.77f32, 90.26f32, 60.41f32, 3.86f32, 61.21f32, 16.6f32, 54.46f32,
            7.29f32, 54.24f32, 92.49f32, 30.18f32, 65.36f32, 99.09f32, 3.8f32, 36.4f32, 86.72f32,
            65.18f32, 29.87f32, 62.21f32, 58.32f32, 43.23f32, 94.3f32, 79.61f32, 39.67f32,
            11.18f32, 48.88f32, 38.19f32, 93.95f32, 10.46f32, 36.7f32, 14.75f32, 81.64f32,
            59.18f32, 99.03f32, 74.23f32, 1.26f32, 82.69f32, 35.7f32, 38.39f32, 46.17f32, 64.75f32,
            7.15f32, 36.55f32, 77.32f32, 18.65f32, 32.8f32, 74.84f32, 18.12f32, 20.19f32, 70.06f32,
            48.37f32, 40.18f32, 45.69f32, 88.3f32, 39.15f32, 60.97f32, 71.29f32, 61.79f32,
            47.23f32, 94.71f32, 58.04f32, 52.4f32, 34.66f32, 59.1f32, 47.11f32, 30.2f32, 58.72f32,
            74.35f32, 83.68f32, 66.8f32, 28.57f32, 29.45f32, 52.02f32, 91.95f32, 92.44f32,
            65.25f32, 38.3f32, 35.6f32, 41.67f32, 91.33f32, 76.81f32, 74.88f32, 33.17f32, 48.36f32,
            41.42f32, 23f32, 8.31f32, 81.69f32, 80.08f32, 50.55f32, 54.46f32, 23.79f32, 43.46f32,
            84.5f32, 10.42f32, 29.51f32, 19.73f32, 46.48f32, 35.01f32, 52.3f32, 66.97f32, 4.8f32,
            74.81f32, 2.82f32, 61.82f32, 25.06f32, 17.3f32, 17.29f32, 63.2f32, 64.1f32, 61.68f32,
            37.42f32, 3.39f32, 97.45f32, 5.32f32, 59.02f32, 35.6f32,
        ];
        fixed_chunk_pq_table.preprocess_query(&mut query_vec);
        assert_eq!(query_vec[0], 32.39f32 - fixed_chunk_pq_table.centroids[0]);
        assert_eq!(
            query_vec[127],
            35.6f32 - fixed_chunk_pq_table.centroids[127]
        );
    }

    #[test]
    fn calculate_distances_tests() {
        let pq_pivots_path: &str = "tests/data/siftsmall_learn.bin_pq_pivots.bin";

        let (dim, pq_table, centroids, chunk_offsets) =
            load_pq_pivots_bin(pq_pivots_path, &1).unwrap();
        let fixed_chunk_pq_table =
            FixedChunkPQTable::new(dim, 1, pq_table, centroids, chunk_offsets);

        let query_vec: Vec<f32> = vec![
            32.39f32, 78.57f32, 50.32f32, 80.46f32, 6.47f32, 69.76f32, 94.2f32, 83.36f32, 5.8f32,
            68.78f32, 42.32f32, 61.77f32, 90.26f32, 60.41f32, 3.86f32, 61.21f32, 16.6f32, 54.46f32,
            7.29f32, 54.24f32, 92.49f32, 30.18f32, 65.36f32, 99.09f32, 3.8f32, 36.4f32, 86.72f32,
            65.18f32, 29.87f32, 62.21f32, 58.32f32, 43.23f32, 94.3f32, 79.61f32, 39.67f32,
            11.18f32, 48.88f32, 38.19f32, 93.95f32, 10.46f32, 36.7f32, 14.75f32, 81.64f32,
            59.18f32, 99.03f32, 74.23f32, 1.26f32, 82.69f32, 35.7f32, 38.39f32, 46.17f32, 64.75f32,
            7.15f32, 36.55f32, 77.32f32, 18.65f32, 32.8f32, 74.84f32, 18.12f32, 20.19f32, 70.06f32,
            48.37f32, 40.18f32, 45.69f32, 88.3f32, 39.15f32, 60.97f32, 71.29f32, 61.79f32,
            47.23f32, 94.71f32, 58.04f32, 52.4f32, 34.66f32, 59.1f32, 47.11f32, 30.2f32, 58.72f32,
            74.35f32, 83.68f32, 66.8f32, 28.57f32, 29.45f32, 52.02f32, 91.95f32, 92.44f32,
            65.25f32, 38.3f32, 35.6f32, 41.67f32, 91.33f32, 76.81f32, 74.88f32, 33.17f32, 48.36f32,
            41.42f32, 23f32, 8.31f32, 81.69f32, 80.08f32, 50.55f32, 54.46f32, 23.79f32, 43.46f32,
            84.5f32, 10.42f32, 29.51f32, 19.73f32, 46.48f32, 35.01f32, 52.3f32, 66.97f32, 4.8f32,
            74.81f32, 2.82f32, 61.82f32, 25.06f32, 17.3f32, 17.29f32, 63.2f32, 64.1f32, 61.68f32,
            37.42f32, 3.39f32, 97.45f32, 5.32f32, 59.02f32, 35.6f32,
        ];

        let dist_vec = fixed_chunk_pq_table.populate_chunk_distances(&query_vec);
        assert_eq!(dist_vec.len(), 256);

        // populate_chunk_distances_test
        let mut sampled_output = 0.0;
        (0..DIM).for_each(|dim_offset| {
            let diff = fixed_chunk_pq_table.pq_table[dim_offset] - query_vec[dim_offset];
            sampled_output += diff * diff;
        });
        assert_eq!(sampled_output, dist_vec[0]);

        // populate_chunk_inner_products_test
        let dist_vec = fixed_chunk_pq_table.populate_chunk_inner_products(&query_vec);
        assert_eq!(dist_vec.len(), 256);

        let mut sampled_output = 0.0;
        (0..DIM).for_each(|dim_offset| {
            sampled_output -= fixed_chunk_pq_table.pq_table[dim_offset] * query_vec[dim_offset];
        });
        assert_eq!(sampled_output, dist_vec[0]);

        // l2_distance_test
        let base_vec: Vec<u8> = vec![3u8];
        let dist = fixed_chunk_pq_table.l2_distance(&query_vec, &base_vec);
        let mut l2_output = 0.0;
        (0..DIM).for_each(|dim_offset| {
            let diff = fixed_chunk_pq_table.pq_table[3 * DIM + dim_offset] - query_vec[dim_offset];
            l2_output += diff * diff;
        });
        assert_eq!(l2_output, dist);

        // inner_product_test
        let dist = fixed_chunk_pq_table.inner_product(&query_vec, &base_vec);
        let mut l2_output = 0.0;
        (0..DIM).for_each(|dim_offset| {
            l2_output -=
                fixed_chunk_pq_table.pq_table[3 * DIM + dim_offset] * query_vec[dim_offset];
        });
        assert_eq!(l2_output, dist);

        // inflate_vector_test
        let inflate_vector = fixed_chunk_pq_table.inflate_vector(&base_vec).unwrap();
        assert_eq!(inflate_vector.len(), DIM);
        assert_eq!(
            inflate_vector[0],
            fixed_chunk_pq_table.pq_table[3 * DIM] + fixed_chunk_pq_table.centroids[0]
        );
        assert_eq!(
            inflate_vector[1],
            fixed_chunk_pq_table.pq_table[3 * DIM + 1] + fixed_chunk_pq_table.centroids[1]
        );
        assert_eq!(
            inflate_vector[127],
            fixed_chunk_pq_table.pq_table[3 * DIM + 127] + fixed_chunk_pq_table.centroids[127]
        );
    }

    fn load_pq_pivots_bin(
        pq_pivots_path: &str,
        num_pq_chunks: &usize,
    ) -> ANNResult<(usize, Vec<f32>, Vec<f32>, Vec<usize>)> {
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

        let (data, pq_center_num, dim) = load_bin::<f32>(pq_pivots_path, file_offset_data[0])?;
        let pq_table = data.to_vec();
        if pq_center_num != NUM_PQ_CENTROIDS {
            let error_message = format!(
                "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                pq_pivots_path, pq_center_num, NUM_PQ_CENTROIDS
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

        Ok((dim, pq_table, centroids, chunk_offsets))
    }
}

#[cfg(test)]
mod pq_index_prune_query_test {

    use super::*;

    #[test]
    fn pq_dist_lookup_test() {
        let pq_ids: Vec<u8> = vec![1u8, 3u8, 2u8, 2u8];
        let mut pq_dists: Vec<f32> = Vec::with_capacity(256 * 2);
        for _ in 0..pq_dists.capacity() {
            pq_dists.push(rand::random());
        }

        let dists_out = pq_dist_lookup(&pq_ids, 2, 2, &pq_dists);
        assert_eq!(dists_out.len(), 2);
        assert_eq!(dists_out[0], pq_dists[0 + 1] + pq_dists[256 + 3]);
        assert_eq!(dists_out[1], pq_dists[0 + 2] + pq_dists[256 + 2]);
    }
}
