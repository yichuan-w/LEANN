/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations)]

use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use crate::common::{ANNError, ANNResult};
use crate::storage::PQStorage;
use crate::utils::{compute_closest_centers, file_exists, k_means_clustering};

/// Max size of PQ training set
pub const MAX_PQ_TRAINING_SET_SIZE: f64 = 256_000f64;

/// Max number of PQ chunks
pub const MAX_PQ_CHUNKS: usize = 512;

pub const NUM_PQ_CENTROIDS: usize = 256;
/// block size for reading/processing large files and matrices in blocks
const BLOCK_SIZE: usize = 5000000;
const NUM_KMEANS_REPS_PQ: usize = 12;

/// given training data in train_data of dimensions num_train * dim, generate
/// PQ pivots using k-means algorithm to partition the co-ordinates into
/// num_pq_chunks (if it divides dimension, else rounded) chunks, and runs
/// k-means in each chunk to compute the PQ pivots and stores in bin format in
/// file pq_pivots_path as a s num_centers*dim floating point binary file
/// PQ pivot table layout: {pivot offsets data: METADATA_SIZE}{pivot vector:[dim; num_centroid]}{centroid vector:[dim; 1]}{chunk offsets:[chunk_num+1; 1]}
fn generate_pq_pivots(
    train_data: &mut [f32],
    num_train: usize,
    dim: usize,
    num_centers: usize,
    num_pq_chunks: usize,
    max_k_means_reps: usize,
    pq_storage: &mut PQStorage,
) -> ANNResult<()> {
    if num_pq_chunks > dim {
        return Err(ANNError::log_pq_error(
            "Error: number of chunks more than dimension.".to_string(),
        ));
    }

    if pq_storage.pivot_data_exist() {
        let (file_num_centers, file_dim) = pq_storage.read_pivot_metadata()?;
        if file_dim == dim && file_num_centers == num_centers {
            // PQ pivot file exists. Not generating again.
            return Ok(());
        }
    }

    // Calculate centroid and center the training data
    // If we use L2 distance, there is an option to
    // translate all vectors to make them centered and
    // then compute PQ. This needs to be set to false
    // when using PQ for MIPS as such translations dont
    // preserve inner products.
    // Now, we're using L2 as default.
    let mut centroid: Vec<f32> = vec![0.0; dim];
    for dim_index in 0..dim {
        for train_data_index in 0..num_train {
            centroid[dim_index] += train_data[train_data_index * dim + dim_index];
        }
        centroid[dim_index] /= num_train as f32;
    }
    for dim_index in 0..dim {
        for train_data_index in 0..num_train {
            train_data[train_data_index * dim + dim_index] -= centroid[dim_index];
        }
    }

    // Calculate each chunk's offset
    // If we have 8 dimension and 3 chunk then offsets would be [0,3,6,8]
    let mut chunk_offsets: Vec<usize> = vec![0; num_pq_chunks + 1];
    let mut chunk_offset: usize = 0;
    for chunk_index in 0..num_pq_chunks {
        chunk_offset += dim / num_pq_chunks;
        if chunk_index < (dim % num_pq_chunks) {
            chunk_offset += 1;
        }
        chunk_offsets[chunk_index + 1] = chunk_offset;
    }

    let mut full_pivot_data: Vec<f32> = vec![0.0; num_centers * dim];
    for chunk_index in 0..num_pq_chunks {
        let chunk_size = chunk_offsets[chunk_index + 1] - chunk_offsets[chunk_index];

        let mut cur_train_data: Vec<f32> = vec![0.0; num_train * chunk_size];
        let mut cur_pivot_data: Vec<f32> = vec![0.0; num_centers * chunk_size];

        cur_train_data
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(train_data_index, chunk)| {
                for (dim_offset, item) in chunk.iter_mut().enumerate() {
                    *item = train_data
                        [train_data_index * dim + chunk_offsets[chunk_index] + dim_offset];
                }
            });

        // Run kmeans to get the centroids of this chunk.
        let (_closest_docs, _closest_center, _residual) = k_means_clustering(
            &cur_train_data,
            num_train,
            chunk_size,
            &mut cur_pivot_data,
            num_centers,
            max_k_means_reps,
        )?;

        // Copy centroids from this chunk table to full table
        for center_index in 0..num_centers {
            full_pivot_data[center_index * dim + chunk_offsets[chunk_index]
                ..center_index * dim + chunk_offsets[chunk_index + 1]]
                .copy_from_slice(
                    &cur_pivot_data[center_index * chunk_size..(center_index + 1) * chunk_size],
                );
        }
    }

    pq_storage.write_pivot_data(
        &full_pivot_data,
        &centroid,
        &chunk_offsets,
        num_centers,
        dim,
    )?;

    Ok(())
}

/// streams the base file (data_file), and computes the closest centers in each
/// chunk to generate the compressed data_file and stores it in
/// pq_compressed_vectors_path.
/// If the numbber of centers is < 256, it stores as byte vector, else as
/// 4-byte vector in binary format.
/// Compressed PQ table layout: {num_points: usize}{num_chunks: usize}{compressed pq table: [num_points; num_chunks]}
fn generate_pq_data_from_pivots<T: Copy + Into<f32>>(
    num_centers: usize,
    num_pq_chunks: usize,
    pq_storage: &mut PQStorage,
) -> ANNResult<()> {
    let (num_points, dim) = pq_storage.read_pq_data_metadata()?;

    let full_pivot_data: Vec<f32>;
    let centroid: Vec<f32>;
    let chunk_offsets: Vec<usize>;

    if !pq_storage.pivot_data_exist() {
        return Err(ANNError::log_pq_error(
            "ERROR: PQ k-means pivot file not found.".to_string(),
        ));
    } else {
        (full_pivot_data, centroid, chunk_offsets) =
            pq_storage.load_pivot_data(&num_pq_chunks, &num_centers, &dim)?;
    }

    pq_storage.write_compressed_pivot_metadata(num_points as i32, num_pq_chunks as i32)?;

    let block_size = if num_points <= BLOCK_SIZE {
        num_points
    } else {
        BLOCK_SIZE
    };
    let num_blocks = (num_points / block_size) + (num_points % block_size != 0) as usize;

    for block_index in 0..num_blocks {
        let start_index: usize = block_index * block_size;
        let end_index: usize = std::cmp::min((block_index + 1) * block_size, num_points);
        let cur_block_size: usize = end_index - start_index;

        let mut block_compressed_base: Vec<usize> = vec![0; cur_block_size * num_pq_chunks];

        let block_data: Vec<T> = pq_storage.read_pq_block_data(cur_block_size, dim)?;

        let mut adjusted_block_data: Vec<f32> = vec![0.0; cur_block_size * dim];

        for block_data_index in 0..cur_block_size {
            for dim_index in 0..dim {
                adjusted_block_data[block_data_index * dim + dim_index] =
                    block_data[block_data_index * dim + dim_index].into() - centroid[dim_index];
            }
        }

        for chunk_index in 0..num_pq_chunks {
            let cur_chunk_size = chunk_offsets[chunk_index + 1] - chunk_offsets[chunk_index];
            if cur_chunk_size == 0 {
                continue;
            }

            let mut cur_pivot_data: Vec<f32> = vec![0.0; num_centers * cur_chunk_size];
            let mut cur_data: Vec<f32> = vec![0.0; cur_block_size * cur_chunk_size];
            let mut closest_center: Vec<u32> = vec![0; cur_block_size];

            // Divide the data into chunks and process each chunk in parallel.
            cur_data
                .par_chunks_mut(cur_chunk_size)
                .enumerate()
                .for_each(|(block_data_index, chunk)| {
                    for (dim_offset, item) in chunk.iter_mut().enumerate() {
                        *item = adjusted_block_data
                            [block_data_index * dim + chunk_offsets[chunk_index] + dim_offset];
                    }
                });

            cur_pivot_data
                .par_chunks_mut(cur_chunk_size)
                .enumerate()
                .for_each(|(center_index, chunk)| {
                    for (din_offset, item) in chunk.iter_mut().enumerate() {
                        *item = full_pivot_data
                            [center_index * dim + chunk_offsets[chunk_index] + din_offset];
                    }
                });

            // Compute the closet centers
            compute_closest_centers(
                &cur_data,
                cur_block_size,
                cur_chunk_size,
                &cur_pivot_data,
                num_centers,
                1,
                &mut closest_center,
                None,
                None,
            )?;

            block_compressed_base
                .par_chunks_mut(num_pq_chunks)
                .enumerate()
                .for_each(|(block_data_index, slice)| {
                    slice[chunk_index] = closest_center[block_data_index] as usize;
                });
        }

        _ = pq_storage.write_compressed_pivot_data(
            &block_compressed_base,
            num_centers,
            cur_block_size,
            num_pq_chunks,
        );
    }
    Ok(())
}

/// Save the data on a file.
/// # Arguments
/// * `p_val` - choose how many ratio sample data as trained data to get pivot
/// * `num_pq_chunks` - pq chunk number
/// * `codebook_prefix` - predefined pivots file named
/// * `pq_storage` - pq file access
pub fn generate_quantized_data<T: Default + Copy + Into<f32>>(
    p_val: f64,
    num_pq_chunks: usize,
    codebook_prefix: &str,
    pq_storage: &mut PQStorage,
) -> ANNResult<()> {
    // If predefined pivots already exists, skip training.
    if !file_exists(codebook_prefix) {
        // Instantiates train data with random sample updates train_data_vector
        // Training data with train_size samples loaded.
        // Each sampled file has train_dim.
        let (mut train_data_vector, train_size, train_dim) =
            pq_storage.gen_random_slice::<T>(p_val)?;

        generate_pq_pivots(
            &mut train_data_vector,
            train_size,
            train_dim,
            NUM_PQ_CENTROIDS,
            num_pq_chunks,
            NUM_KMEANS_REPS_PQ,
            pq_storage,
        )?;
    }
    generate_pq_data_from_pivots::<T>(NUM_PQ_CENTROIDS, num_pq_chunks, pq_storage)?;
    Ok(())
}

#[cfg(test)]
mod pq_test {

    use std::fs::File;
    use std::io::Write;

    use super::*;
    use crate::utils::{convert_types_u32_usize, convert_types_u64_usize, load_bin, METADATA_SIZE};

    #[test]
    fn generate_pq_pivots_test() {
        let pivot_file_name = "generate_pq_pivots_test.bin";
        let compressed_file_name = "compressed.bin";
        let pq_training_file_name = "tests/data/siftsmall_learn.bin";
        let mut pq_storage =
            PQStorage::new(pivot_file_name, compressed_file_name, pq_training_file_name).unwrap();
        let mut train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];
        generate_pq_pivots(&mut train_data, 5, 8, 2, 2, 5, &mut pq_storage).unwrap();

        let (data, nr, nc) = load_bin::<u64>(pivot_file_name, 0).unwrap();
        let file_offset_data = convert_types_u64_usize(&data, nr, nc);
        assert_eq!(file_offset_data[0], METADATA_SIZE);
        assert_eq!(nr, 4);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<f32>(pivot_file_name, file_offset_data[0]).unwrap();
        let full_pivot_data = data.to_vec();
        assert_eq!(full_pivot_data.len(), 16);
        assert_eq!(nr, 2);
        assert_eq!(nc, 8);

        let (data, nr, nc) = load_bin::<f32>(pivot_file_name, file_offset_data[1]).unwrap();
        let centroid = data.to_vec();
        assert_eq!(
            centroid[0],
            (1.0f32 + 2.0f32 + 2.1f32 + 2.2f32 + 100.0f32) / 5.0f32
        );
        assert_eq!(nr, 8);
        assert_eq!(nc, 1);

        let (data, nr, nc) = load_bin::<u32>(pivot_file_name, file_offset_data[2]).unwrap();
        let chunk_offsets = convert_types_u32_usize(&data, nr, nc);
        assert_eq!(chunk_offsets[0], 0);
        assert_eq!(chunk_offsets[1], 4);
        assert_eq!(chunk_offsets[2], 8);
        assert_eq!(nr, 3);
        assert_eq!(nc, 1);
        std::fs::remove_file(pivot_file_name).unwrap();
    }

    #[test]
    fn generate_pq_data_from_pivots_test() {
        let data_file = "generate_pq_data_from_pivots_test_data.bin";
        //npoints=5, dim=8, 5 vectors [1.0;8] [2.0;8] [2.1;8] [2.2;8] [100.0;8]
        let mut train_data: Vec<f32> = vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
            2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32,
            100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
        ];
        let my_nums_unstructured: &[u8] = unsafe {
            std::slice::from_raw_parts(train_data.as_ptr() as *const u8, train_data.len() * 4)
        };
        let meta: Vec<i32> = vec![5, 8];
        let meta_unstructured: &[u8] =
            unsafe { std::slice::from_raw_parts(meta.as_ptr() as *const u8, meta.len() * 4) };
        let mut data_file_writer = File::create(data_file).unwrap();
        data_file_writer
            .write_all(meta_unstructured)
            .expect("Failed to write sample file");
        data_file_writer
            .write_all(my_nums_unstructured)
            .expect("Failed to write sample file");

        let pq_pivots_path = "generate_pq_data_from_pivots_test_pivot.bin";
        let pq_compressed_vectors_path = "generate_pq_data_from_pivots_test.bin";
        let mut pq_storage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, data_file).unwrap();
        generate_pq_pivots(&mut train_data, 5, 8, 2, 2, 5, &mut pq_storage).unwrap();
        generate_pq_data_from_pivots::<f32>(2, 2, &mut pq_storage).unwrap();
        let (data, nr, nc) = load_bin::<u8>(pq_compressed_vectors_path, 0).unwrap();
        assert_eq!(nr, 5);
        assert_eq!(nc, 2);
        assert_eq!(data[0], data[2]);
        assert_ne!(data[0], data[8]);

        std::fs::remove_file(data_file).unwrap();
        std::fs::remove_file(pq_pivots_path).unwrap();
        std::fs::remove_file(pq_compressed_vectors_path).unwrap();
    }

    #[test]
    fn pq_end_to_end_validation_with_codebook_test() {
        let data_file = "tests/data/siftsmall_learn.bin";
        let pq_pivots_path = "tests/data/siftsmall_learn.bin_pq_pivots.bin";
        let gound_truth_path = "tests/data/siftsmall_learn.bin_pq_compressed.bin";
        let pq_compressed_vectors_path = "validation.bin";
        let mut pq_storage =
            PQStorage::new(pq_pivots_path, pq_compressed_vectors_path, data_file).unwrap();
        generate_quantized_data::<f32>(0.5, 1, pq_pivots_path, &mut pq_storage).unwrap();

        let (data, nr, nc) = load_bin::<u8>(pq_compressed_vectors_path, 0).unwrap();
        let (gt_data, gt_nr, gt_nc) = load_bin::<u8>(gound_truth_path, 0).unwrap();
        assert_eq!(nr, gt_nr);
        assert_eq!(nc, gt_nc);
        for i in 0..data.len() {
            assert_eq!(data[i], gt_data[i]);
        }
        std::fs::remove_file(pq_compressed_vectors_path).unwrap();
    }
}
