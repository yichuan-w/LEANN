/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use bytemuck::{cast_slice, Pod};
use diskann::{
    common::{ANNError, ANNResult, AlignedBoxWithSlice},
    model::data_store::DatasetDto,
    utils::{copy_aligned_data_from_file, is_aligned, round_up},
};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;

pub(crate) fn calculate_recall(
    num_queries: usize,
    gold_std: &[u32],
    gs_dist: &Option<Vec<f32>>,
    dim_gs: usize,
    our_results: &[u32],
    dim_or: u32,
    recall_at: u32,
) -> ANNResult<f64> {
    let mut total_recall: f64 = 0.0;
    let (mut gt, mut res): (HashSet<u32>, HashSet<u32>) = (HashSet::new(), HashSet::new());

    for i in 0..num_queries {
        gt.clear();
        res.clear();

        let gt_slice = &gold_std[dim_gs * i..];
        let res_slice = &our_results[dim_or as usize * i..];
        let mut tie_breaker = recall_at as usize;

        if gs_dist.is_some() {
            tie_breaker = (recall_at - 1) as usize;
            let gt_dist_vec = &gs_dist.as_ref().unwrap()[dim_gs * i..];
            while tie_breaker < dim_gs
                && gt_dist_vec[tie_breaker] == gt_dist_vec[(recall_at - 1) as usize]
            {
                tie_breaker += 1;
            }
        }

        (0..tie_breaker).for_each(|idx| {
            gt.insert(gt_slice[idx]);
        });

        (0..tie_breaker).for_each(|idx| {
            res.insert(res_slice[idx]);
        });

        let mut cur_recall: u32 = 0;
        for v in gt.iter() {
            if res.contains(v) {
                cur_recall += 1;
            }
        }

        total_recall += cur_recall as f64;
    }

    Ok(total_recall / num_queries as f64 * (100.0 / recall_at as f64))
}

pub(crate) fn get_graph_num_frozen_points(graph_file: &str) -> ANNResult<usize> {
    let mut file = File::open(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut usize_buffer)?;
    let file_frozen_pts = usize::from_le_bytes(usize_buffer);

    Ok(file_frozen_pts)
}

#[inline]
pub(crate) fn load_truthset(
    bin_file: &str,
) -> ANNResult<(Vec<u32>, Option<Vec<f32>>, usize, usize)> {
    let mut file = File::open(bin_file)?;
    let actual_file_size = file.metadata()?.len() as usize;

    let mut buffer = [0; size_of::<i32>()];
    file.read_exact(&mut buffer)?;
    let npts = i32::from_le_bytes(buffer) as usize;

    file.read_exact(&mut buffer)?;
    let dim = i32::from_le_bytes(buffer) as usize;

    println!("Metadata: #pts = {npts}, #dims = {dim}... ");

    let expected_file_size_with_dists: usize =
        2 * npts * dim * size_of::<u32>() + 2 * size_of::<u32>();
    let expected_file_size_just_ids: usize = npts * dim * size_of::<u32>() + 2 * size_of::<u32>();

    let truthset_type : i32 = match actual_file_size
    {
        // This is in the C++ code, but nothing is done in this case. Keeping it here for future reference just in case.
        // expected_file_size_just_ids => 2,
        x if x == expected_file_size_with_dists => 1,
        _ => return Err(ANNError::log_index_error(format!("Error. File size mismatch. File should have bin format, with npts followed by ngt 
                                                        followed by npts*ngt ids and optionally followed by npts*ngt distance values; actual size: {}, expected: {} or {}",
                                                        actual_file_size,
                                                        expected_file_size_with_dists,
                                                        expected_file_size_just_ids)))
    };

    let mut ids: Vec<u32> = vec![0; npts * dim];
    let mut buffer = vec![0; npts * dim * size_of::<u32>()];
    file.read_exact(&mut buffer)?;
    ids.clone_from_slice(cast_slice::<u8, u32>(&buffer));

    if truthset_type == 1 {
        let mut dists: Vec<f32> = vec![0.0; npts * dim];
        let mut buffer = vec![0; npts * dim * size_of::<f32>()];
        file.read_exact(&mut buffer)?;
        dists.clone_from_slice(cast_slice::<u8, f32>(&buffer));

        return Ok((ids, Some(dists), npts, dim));
    }

    Ok((ids, None, npts, dim))
}

#[inline]
pub(crate) fn load_aligned_bin<T: Default + Copy + Sized + Pod>(
    bin_file: &str,
) -> ANNResult<(AlignedBoxWithSlice<T>, usize, usize, usize)> {
    let t_size = size_of::<T>();
    let (npts, dim, file_size): (usize, usize, usize);
    {
        println!("Reading (with alignment) bin file: {bin_file}");
        let mut file = File::open(bin_file)?;
        file_size = file.metadata()?.len() as usize;

        let mut buffer = [0; size_of::<i32>()];
        file.read_exact(&mut buffer)?;
        npts = i32::from_le_bytes(buffer) as usize;

        file.read_exact(&mut buffer)?;
        dim = i32::from_le_bytes(buffer) as usize;
    }

    let rounded_dim = round_up(dim, 8);
    let expected_actual_file_size = npts * dim * size_of::<T>() + 2 * size_of::<u32>();

    if file_size != expected_actual_file_size {
        return Err(ANNError::log_index_error(format!(
            "ERROR: File size mismatch. Actual size is {} while expected size is {} 
        npts = {}, #dims = {}, aligned_dim = {}",
            file_size, expected_actual_file_size, npts, dim, rounded_dim
        )));
    }

    println!("Metadata: #pts = {npts}, #dims = {dim}, aligned_dim = {rounded_dim}...");

    let alloc_size = npts * rounded_dim;
    let alignment = 8 * t_size;
    println!(
        "allocating aligned memory of {} bytes... ",
        alloc_size * t_size
    );
    if !is_aligned(alloc_size * t_size, alignment) {
        return Err(ANNError::log_index_error(format!(
            "Requested memory size is not a multiple of {}. Can not be allocated.",
            alignment
        )));
    }

    let mut data = AlignedBoxWithSlice::<T>::new(alloc_size, alignment)?;
    let dto = DatasetDto {
        data: &mut data,
        rounded_dim,
    };

    println!("done. Copying data to mem_aligned buffer...");

    let (_, _) = copy_aligned_data_from_file(bin_file, dto, 0)?;

    Ok((data, npts, dim, rounded_dim))
}
