/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

extern crate cblas;
extern crate openblas_src;

use cblas::{sgemm, snrm2, Layout, Transpose};
use rayon::prelude::*;
use std::{
    cmp::{min, Ordering},
    collections::BinaryHeap,
    sync::{Arc, Mutex},
};

use crate::common::{ANNError, ANNResult};

struct PivotContainer {
    piv_id: usize,
    piv_dist: f32,
}

impl PartialOrd for PivotContainer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.piv_dist.partial_cmp(&self.piv_dist)
    }
}

impl Ord for PivotContainer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Treat NaN as less than all other values.
        // piv_dist should never be NaN.
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

impl PartialEq for PivotContainer {
    fn eq(&self, other: &Self) -> bool {
        self.piv_dist == other.piv_dist
    }
}

impl Eq for PivotContainer {}

/// Calculate the Euclidean distance between two vectors
pub fn calc_distance(vec_1: &[f32], vec_2: &[f32], dim: usize) -> f32 {
    let mut dist = 0.0;
    for j in 0..dim {
        let diff = vec_1[j] - vec_2[j];
        dist += diff * diff;
    }
    dist
}

/// Compute L2-squared norms of data stored in row-major num_points * dim,
/// need to be pre-allocated
pub fn compute_vecs_l2sq(vecs_l2sq: &mut [f32], data: &[f32], num_points: usize, dim: usize) {
    assert_eq!(vecs_l2sq.len(), num_points);

    vecs_l2sq
        .par_iter_mut()
        .enumerate()
        .for_each(|(n_iter, vec_l2sq)| {
            let slice = &data[n_iter * dim..(n_iter + 1) * dim];
            let norm = unsafe { snrm2(dim as i32, slice, 1) };
            *vec_l2sq = norm * norm;
        });
}

/// Calculate k closest centers to data of num_points * dim (row-major)
/// Centers is num_centers * dim (row-major)
/// data_l2sq has pre-computed squared norms of data
/// centers_l2sq has pre-computed squared norms of centers
/// Pre-allocated center_index will contain id of nearest center
/// Pre-allocated dist_matrix should be num_points * num_centers and contain squared distances
/// Default value of k is 1
/// Ideally used only by compute_closest_centers
#[allow(clippy::too_many_arguments)]
pub fn compute_closest_centers_in_block(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &[f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    centers_l2sq: &[f32],
    center_index: &mut [u32],
    dist_matrix: &mut [f32],
    k: usize,
) -> ANNResult<()> {
    if k > num_centers {
        return Err(ANNError::log_index_error(format!(
            "ERROR: k ({}) > num_centers({})",
            k, num_centers
        )));
    }

    let ones_a: Vec<f32> = vec![1.0; num_centers];
    let ones_b: Vec<f32> = vec![1.0; num_points];

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            num_points as i32,
            num_centers as i32,
            1,
            1.0,
            docs_l2sq,
            1,
            &ones_a,
            1,
            0.0,
            dist_matrix,
            num_centers as i32,
        );
    }

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            num_points as i32,
            num_centers as i32,
            1,
            1.0,
            &ones_b,
            1,
            centers_l2sq,
            1,
            1.0,
            dist_matrix,
            num_centers as i32,
        );
    }

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            num_points as i32,
            num_centers as i32,
            dim as i32,
            -2.0,
            data,
            dim as i32,
            centers,
            dim as i32,
            1.0,
            dist_matrix,
            num_centers as i32,
        );
    }

    if k == 1 {
        center_index
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, center_idx)| {
                let mut min = f32::MAX;
                let current = &dist_matrix[i * num_centers..(i + 1) * num_centers];
                let mut min_idx = 0;
                for (j, &distance) in current.iter().enumerate() {
                    if distance < min {
                        min = distance;
                        min_idx = j;
                    }
                }
                *center_idx = min_idx as u32;
            });
    } else {
        center_index
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, center_chunk)| {
                let current = &dist_matrix[i * num_centers..(i + 1) * num_centers];
                let mut top_k_queue = BinaryHeap::new();
                for (j, &distance) in current.iter().enumerate() {
                    let this_piv = PivotContainer {
                        piv_id: j,
                        piv_dist: distance,
                    };
                    if top_k_queue.len() < k {
                        top_k_queue.push(this_piv);
                    } else {
                        // Safe unwrap, top_k_queue is not empty
                        #[allow(clippy::unwrap_used)]
                        let mut top = top_k_queue.peek_mut().unwrap();
                        if this_piv.piv_dist < top.piv_dist {
                            *top = this_piv;
                        }
                    }
                }
                for (_j, center_idx) in center_chunk.iter_mut().enumerate() {
                    if let Some(this_piv) = top_k_queue.pop() {
                        *center_idx = this_piv.piv_id as u32;
                    } else {
                        break;
                    }
                }
            });
    }

    Ok(())
}

/// Given data in num_points * new_dim row major
/// Pivots stored in full_pivot_data as num_centers * new_dim row major
/// Calculate the k closest pivot for each point and store it in vector
/// closest_centers_ivf (row major, num_points*k) (which needs to be allocated
/// outside) Additionally, if inverted index is not null (and pre-allocated),
/// it will return inverted index for each center, assuming each of the inverted
/// indices is an empty vector. Additionally, if pts_norms_squared is not null,
/// then it will assume that point norms are pre-computed and use those values
#[allow(clippy::too_many_arguments)]
pub fn compute_closest_centers(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &[f32],
    num_centers: usize,
    k: usize,
    closest_centers_ivf: &mut [u32],
    mut inverted_index: Option<&mut Vec<Vec<usize>>>,
    pts_norms_squared: Option<&[f32]>,
) -> ANNResult<()> {
    if k > num_centers {
        return Err(ANNError::log_index_error(format!(
            "ERROR: k ({}) > num_centers({})",
            k, num_centers
        )));
    }

    let _is_norm_given_for_pts = pts_norms_squared.is_some();

    let mut pivs_norms_squared = vec![0.0; num_centers];

    let mut pts_norms_squared = if let Some(pts_norms) = pts_norms_squared {
        pts_norms.to_vec()
    } else {
        let mut norms_squared = vec![0.0; num_points];
        compute_vecs_l2sq(&mut norms_squared, data, num_points, dim);
        norms_squared
    };

    compute_vecs_l2sq(&mut pivs_norms_squared, pivot_data, num_centers, dim);

    let par_block_size = num_points;
    let n_blocks = if num_points % par_block_size == 0 {
        num_points / par_block_size
    } else {
        num_points / par_block_size + 1
    };

    let mut closest_centers = vec![0u32; par_block_size * k];
    let mut distance_matrix = vec![0.0; num_centers * par_block_size];

    for cur_blk in 0..n_blocks {
        let data_cur_blk = &data[cur_blk * par_block_size * dim..];
        let num_pts_blk = min(par_block_size, num_points - cur_blk * par_block_size);
        let pts_norms_blk = &mut pts_norms_squared[cur_blk * par_block_size..];

        compute_closest_centers_in_block(
            data_cur_blk,
            num_pts_blk,
            dim,
            pivot_data,
            num_centers,
            pts_norms_blk,
            &pivs_norms_squared,
            &mut closest_centers,
            &mut distance_matrix,
            k,
        )?;

        closest_centers_ivf.clone_from_slice(&closest_centers);

        if let Some(inverted_index_inner) = inverted_index.as_mut() {
            let inverted_index_arc = Arc::new(Mutex::new(inverted_index_inner));

            (0..num_points)
                .into_par_iter()
                .try_for_each(|j| -> ANNResult<()> {
                    let this_center_id = closest_centers[j] as usize;
                    let mut guard = inverted_index_arc.lock().map_err(|err| {
                        ANNError::log_index_error(format!(
                            "PoisonError: Lock poisoned when acquiring inverted_index_arc, err={}",
                            err
                        ))
                    })?;
                    guard[this_center_id].push(j);

                    Ok(())
                })?;
        }
    }

    Ok(())
}

/// If to_subtract is true, will subtract nearest center from each row.
/// Else will add.
/// Output will be in data_load itself.
/// Nearest centers need to be provided in closest_centers.
pub fn process_residuals(
    data_load: &mut [f32],
    num_points: usize,
    dim: usize,
    cur_pivot_data: &[f32],
    num_centers: usize,
    closest_centers: &[u32],
    to_subtract: bool,
) {
    println!(
        "Processing residuals of {} points in {} dimensions using {} centers",
        num_points, dim, num_centers
    );

    data_load
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(n_iter, chunk)| {
            let cur_pivot_index = closest_centers[n_iter] as usize * dim;
            for d_iter in 0..dim {
                if to_subtract {
                    chunk[d_iter] -= cur_pivot_data[cur_pivot_index + d_iter];
                } else {
                    chunk[d_iter] += cur_pivot_data[cur_pivot_index + d_iter];
                }
            }
        });
}

#[cfg(test)]
mod math_util_test {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn calc_distance_test() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let dim = vec1.len();

        let dist = calc_distance(&vec1, &vec2, dim);

        let expected = 27.0;

        assert_eq!(dist, expected);
    }

    #[test]
    fn compute_vecs_l2sq_test() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_points = 2;
        let dim = 3;
        let mut vecs_l2sq = vec![0.0; num_points];

        compute_vecs_l2sq(&mut vecs_l2sq, &data, num_points, dim);

        let expected = vec![14.0, 77.0];

        assert_eq!(vecs_l2sq.len(), num_points);
        assert_abs_diff_eq!(vecs_l2sq[0], expected[0], epsilon = 1e-6);
        assert_abs_diff_eq!(vecs_l2sq[1], expected[1], epsilon = 1e-6);
    }

    #[test]
    fn compute_closest_centers_in_block_test() {
        let num_points = 10;
        let dim = 5;
        let num_centers = 3;
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
        ];
        let centers = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 31.0, 32.0, 33.0, 34.0, 35.0,
        ];
        let mut docs_l2sq = vec![0.0; num_points];
        compute_vecs_l2sq(&mut docs_l2sq, &data, num_points, dim);
        let mut centers_l2sq = vec![0.0; num_centers];
        compute_vecs_l2sq(&mut centers_l2sq, &centers, num_centers, dim);
        let mut center_index = vec![0; num_points];
        let mut dist_matrix = vec![0.0; num_points * num_centers];
        let k = 1;

        compute_closest_centers_in_block(
            &data,
            num_points,
            dim,
            &centers,
            num_centers,
            &docs_l2sq,
            &centers_l2sq,
            &mut center_index,
            &mut dist_matrix,
            k,
        )
        .unwrap();

        assert_eq!(center_index.len(), num_points);
        let expected_center_index = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2];
        assert_abs_diff_eq!(*center_index, expected_center_index);

        assert_eq!(dist_matrix.len(), num_points * num_centers);
        let expected_dist_matrix = vec![
            0.0, 2000.0, 4500.0, 125.0, 1125.0, 3125.0, 500.0, 500.0, 2000.0, 1125.0, 125.0,
            1125.0, 2000.0, 0.0, 500.0, 3125.0, 125.0, 125.0, 4500.0, 500.0, 0.0, 6125.0, 1125.0,
            125.0, 8000.0, 2000.0, 500.0, 10125.0, 3125.0, 1125.0,
        ];
        assert_abs_diff_eq!(*dist_matrix, expected_dist_matrix, epsilon = 1e-2);
    }

    #[test]
    fn test_compute_closest_centers() {
        let num_points = 4;
        let dim = 3;
        let num_centers = 2;
        let mut data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let pivot_data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let k = 1;

        let mut closest_centers_ivf = vec![0u32; num_points * k];
        let mut inverted_index: Vec<Vec<usize>> = vec![vec![], vec![]];

        compute_closest_centers(
            &data,
            num_points,
            dim,
            &pivot_data,
            num_centers,
            k,
            &mut closest_centers_ivf,
            Some(&mut inverted_index),
            None,
        )
        .unwrap();

        assert_eq!(closest_centers_ivf, vec![0, 0, 1, 1]);

        for vec in inverted_index.iter_mut() {
            vec.sort_unstable();
        }
        assert_eq!(inverted_index, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn process_residuals_test() {
        let mut data_load = vec![1.0, 2.0, 3.0, 4.0];
        let num_points = 2;
        let dim = 2;
        let cur_pivot_data = vec![0.5, 1.5, 2.5, 3.5];
        let num_centers = 2;
        let closest_centers = vec![0, 1];
        let to_subtract = true;

        process_residuals(
            &mut data_load,
            num_points,
            dim,
            &cur_pivot_data,
            num_centers,
            &closest_centers,
            to_subtract,
        );

        assert_eq!(data_load, vec![0.5, 0.5, 0.5, 0.5]);
    }
}
