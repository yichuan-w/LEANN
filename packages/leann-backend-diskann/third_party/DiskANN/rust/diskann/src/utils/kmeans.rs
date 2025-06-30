/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use rayon::prelude::*;
use std::cmp::min;

use crate::common::ANNResult;
use crate::utils::math_util::{calc_distance, compute_closest_centers, compute_vecs_l2sq};

/// Run Lloyds one iteration
/// Given data in row-major num_points * dim, and centers in row-major
/// num_centers * dim and squared lengths of ata points, output the closest
/// center to each data point, update centers, and also return inverted index.
/// If closest_centers == NULL, will allocate memory and return.
/// Similarly, if closest_docs == NULL, will allocate memory and return.
#[allow(clippy::too_many_arguments)]
fn lloyds_iter(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    mut closest_docs: &mut Vec<Vec<usize>>,
    closest_center: &mut [u32],
) -> ANNResult<f32> {
    let compute_residual = true;

    closest_docs.iter_mut().for_each(|doc| doc.clear());

    compute_closest_centers(
        data,
        num_points,
        dim,
        centers,
        num_centers,
        1,
        closest_center,
        Some(&mut closest_docs),
        Some(docs_l2sq),
    )?;

    centers.fill(0.0);

    centers
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(c, center)| {
            let mut cluster_sum = vec![0.0; dim];
            for &doc_index in &closest_docs[c] {
                let current = &data[doc_index * dim..(doc_index + 1) * dim];
                for (j, current_val) in current.iter().enumerate() {
                    cluster_sum[j] += *current_val as f64;
                }
            }
            if !closest_docs[c].is_empty() {
                for (i, sum_val) in cluster_sum.iter().enumerate() {
                    center[i] = (*sum_val / closest_docs[c].len() as f64) as f32;
                }
            }
        });

    let mut residual = 0.0;
    if compute_residual {
        let buf_pad: usize = 32;
        let chunk_size: usize = 2 * 8192;
        let nchunks =
            num_points / chunk_size + (if num_points % chunk_size == 0 { 0 } else { 1 } as usize);

        let mut residuals: Vec<f32> = vec![0.0; nchunks * buf_pad];

        residuals
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk, res)| {
                for d in (chunk * chunk_size)..min(num_points, (chunk + 1) * chunk_size) {
                    *res += calc_distance(
                        &data[d * dim..(d + 1) * dim],
                        &centers[closest_center[d] as usize * dim..],
                        dim,
                    );
                }
            });

        for chunk in 0..nchunks {
            residual += residuals[chunk * buf_pad];
        }
    }

    Ok(residual)
}

/// Run Lloyds until max_reps or stopping criterion
/// If you pass NULL for closest_docs and closest_center, it will NOT return
/// the results, else it will assume appropriate allocation as closest_docs =
/// new vec<usize> [num_centers], and closest_center = new size_t[num_points]
/// Final centers are output in centers as row-major num_centers * dim.
fn run_lloyds(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    let mut residual = f32::MAX;

    let mut closest_docs = vec![Vec::new(); num_centers];
    let mut closest_center = vec![0; num_points];

    let mut docs_l2sq = vec![0.0; num_points];
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim);

    let mut old_residual;

    for i in 0..max_reps {
        old_residual = residual;

        residual = lloyds_iter(
            data,
            num_points,
            dim,
            centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
        )?;

        if (i != 0 && (old_residual - residual) / residual < 0.00001) || (residual < f32::EPSILON) {
            println!(
                "Residuals unchanged: {} becomes {}. Early termination.",
                old_residual, residual
            );
            break;
        }
    }

    Ok((closest_docs, closest_center, residual))
}

/// Assume memory allocated for pivot_data as new float[num_centers * dim]
/// and select randomly num_centers points as pivots
fn selecting_pivots(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
) {
    let mut picked = Vec::new();
    let mut rng = thread_rng();
    let distribution = Uniform::from(0..num_points);

    for j in 0..num_centers {
        let mut tmp_pivot = distribution.sample(&mut rng);
        while picked.contains(&tmp_pivot) {
            tmp_pivot = distribution.sample(&mut rng);
        }
        picked.push(tmp_pivot);
        let data_offset = tmp_pivot * dim;
        let pivot_offset = j * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);
    }
}

/// Select pivots in k-means++ algorithm
/// Points that are farther away from the already chosen centroids
/// have a higher probability of being selected as the next centroid.
/// The k-means++ algorithm helps avoid poor initial centroid
/// placement that can result in suboptimal clustering.
fn k_meanspp_selecting_pivots(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
) {
    if num_points > (1 << 23) {
        println!("ERROR: n_pts {} currently not supported for k-means++, maximum is 8388608. Falling back to random pivot selection.", num_points);
        selecting_pivots(data, num_points, dim, pivot_data, num_centers);
        return;
    }

    let mut picked: Vec<usize> = Vec::new();
    let mut rng = thread_rng();
    let real_distribution = Uniform::from(0.0..1.0);
    let int_distribution = Uniform::from(0..num_points);

    let init_id = int_distribution.sample(&mut rng);
    let mut num_picked = 1;

    picked.push(init_id);
    let init_data_offset = init_id * dim;
    pivot_data[0..dim].copy_from_slice(&data[init_data_offset..init_data_offset + dim]);

    let mut dist = vec![0.0; num_points];

    dist.par_iter_mut().enumerate().for_each(|(i, dist_i)| {
        *dist_i = calc_distance(
            &data[i * dim..(i + 1) * dim],
            &data[init_id * dim..(init_id + 1) * dim],
            dim,
        );
    });

    let mut dart_val: f64;
    let mut tmp_pivot = 0;
    let mut sum_flag = false;

    while num_picked < num_centers {
        dart_val = real_distribution.sample(&mut rng);

        let mut sum: f64 = 0.0;
        for item in dist.iter().take(num_points) {
            sum += *item as f64;
        }
        if sum == 0.0 {
            sum_flag = true;
        }

        dart_val *= sum;

        let mut prefix_sum: f64 = 0.0;
        for (i, pivot) in dist.iter().enumerate().take(num_points) {
            tmp_pivot = i;
            if dart_val >= prefix_sum && dart_val < (prefix_sum + *pivot as f64) {
                break;
            }

            prefix_sum += *pivot as f64;
        }

        if picked.contains(&tmp_pivot) && !sum_flag {
            continue;
        }

        picked.push(tmp_pivot);
        let pivot_offset = num_picked * dim;
        let data_offset = tmp_pivot * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);

        dist.par_iter_mut().enumerate().for_each(|(i, dist_i)| {
            *dist_i = (*dist_i).min(calc_distance(
                &data[i * dim..(i + 1) * dim],
                &data[tmp_pivot * dim..(tmp_pivot + 1) * dim],
                dim,
            ));
        });

        num_picked += 1;
    }
}

/// k-means algorithm interface
pub fn k_means_clustering(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    k_meanspp_selecting_pivots(data, num_points, dim, centers, num_centers);
    let (closest_docs, closest_center, residual) =
        run_lloyds(data, num_points, dim, centers, num_centers, max_reps)?;
    Ok((closest_docs, closest_center, residual))
}

#[cfg(test)]
mod kmeans_test {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn lloyds_iter_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];

        let mut closest_docs: Vec<Vec<usize>> = vec![vec![]; num_centers];
        let mut closest_center: Vec<u32> = vec![0; num_points];
        let docs_l2sq: Vec<f32> = data
            .chunks(dim)
            .map(|chunk| chunk.iter().map(|val| val.powi(2)).sum())
            .collect();

        let residual = lloyds_iter(
            &data,
            num_points,
            dim,
            &mut centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
        )
        .unwrap();

        let expected_centers: [f32; 6] = [2.0, 3.0, 9.0, 10.0, 17.0, 18.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![0, 1], vec![2, 3, 4, 5, 6], vec![7, 8, 9]];
        let expected_closest_center: [u32; 10] = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2];
        let expected_residual: f32 = 100.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn run_lloyds_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let max_reps = 5;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];

        let (mut closest_docs, mut closest_center, residual) =
            run_lloyds(&data, num_points, dim, &mut centers, num_centers, max_reps).unwrap();

        let expected_centers: [f32; 6] = [3.0, 4.0, 10.0, 11.0, 17.0, 18.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![0, 1, 2], vec![3, 4, 5, 6], vec![7, 8, 9]];
        let expected_closest_center: [u32; 10] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2];
        let expected_residual: f32 = 72.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn selecting_pivots_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        // Generate some random data points
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.gen()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];

        selecting_pivots(&data, num_points, dim, &mut pivot_data, num_centers);

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..(pivot_offset + dim)];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..(data_offset + dim)];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }

    #[test]
    fn k_meanspp_selecting_pivots_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        // Generate some random data points
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.gen()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];

        k_meanspp_selecting_pivots(&data, num_points, dim, &mut pivot_data, num_centers);

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..pivot_offset + dim];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..data_offset + dim];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }
}
