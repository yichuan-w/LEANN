/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
mod search_index_utils;
use bytemuck::Pod;
use diskann::{
    common::{ANNError, ANNResult},
    index,
    model::{
        configuration::index_write_parameters::{default_param_vals, IndexWriteParametersBuilder},
        vertex::{DIM_104, DIM_128, DIM_256},
        IndexConfiguration,
    },
    utils::{load_metadata_from_file, save_bin_u32},
};
use std::{env, path::Path, process::exit, time::Instant};
use vector::{FullPrecisionDistance, Half, Metric};

use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
fn search_memory_index<T>(
    metric: Metric,
    index_path: &str,
    result_path_prefix: &str,
    query_file: &str,
    truthset_file: &str,
    num_threads: u32,
    recall_at: u32,
    print_all_recalls: bool,
    l_vec: &Vec<u32>,
    show_qps_per_thread: bool,
    fail_if_recall_below: f32,
) -> ANNResult<i32>
where
    T: Default + Copy + Sized + Pod + Sync + Send + Into<f32>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
{
    // Load the query file
    let (query, query_num, query_dim, query_aligned_dim) =
        search_index_utils::load_aligned_bin::<T>(query_file)?;
    let mut gt_dim: usize = 0;
    let mut gt_ids: Option<Vec<u32>> = None;
    let mut gt_dists: Option<Vec<f32>> = None;

    // Check for ground truth
    let mut calc_recall_flag = false;
    if !truthset_file.is_empty() && Path::new(truthset_file).exists() {
        let ret = search_index_utils::load_truthset(truthset_file)?;
        gt_ids = Some(ret.0);
        gt_dists = ret.1;
        let gt_num = ret.2;
        gt_dim = ret.3;

        if gt_num != query_num {
            println!("Error. Mismatch in number of queries and ground truth data");
        }

        calc_recall_flag = true;
    } else {
        println!(
            "Truthset file {} not found. Not computing recall",
            truthset_file
        );
    }

    let num_frozen_pts = search_index_utils::get_graph_num_frozen_points(index_path)?;

    // C++ uses the max given L value, so we do the same here. Max degree is never specified in C++ so use the rust default
    let index_write_params = IndexWriteParametersBuilder::new(
        *l_vec.iter().max().unwrap(),
        default_param_vals::MAX_DEGREE,
    )
    .with_num_threads(num_threads)
    .build();

    let (index_num_points, _) = load_metadata_from_file(&format!("{}.data", index_path))?;

    let index_config = IndexConfiguration::new(
        metric,
        query_dim,
        query_aligned_dim,
        index_num_points,
        false,
        0,
        false,
        num_frozen_pts,
        1f32,
        index_write_params,
    );
    let mut index = index::create_inmem_index::<T>(index_config)?;

    index.load(index_path, index_num_points)?;

    println!("Using {} threads to search", num_threads);
    let qps_title = if show_qps_per_thread {
        "QPS/thread"
    } else {
        "QPS"
    };
    let mut table_width = 4 + 12 + 18 + 20 + 15;
    let mut table_header_str = format!(
        "{:>4}{:>12}{:>18}{:>20}{:>15}",
        "Ls", qps_title, "Avg dist cmps", "Mean Latency (mus)", "99.9 Latency"
    );

    let first_recall: u32 = if print_all_recalls { 1 } else { recall_at };
    let mut recalls_to_print: usize = 0;
    if calc_recall_flag {
        for curr_recall in first_recall..=recall_at {
            let recall_str = format!("Recall@{}", curr_recall);
            table_header_str.push_str(&format!("{:>12}", recall_str));
            recalls_to_print = (recall_at + 1 - first_recall) as usize;
            table_width += recalls_to_print * 12;
        }
    }

    println!("{}", table_header_str);
    println!("{}", "=".repeat(table_width));

    let mut query_result_ids: Vec<Vec<u32>> =
        vec![vec![0; query_num * recall_at as usize]; l_vec.len()];
    let mut latency_stats: Vec<f32> = vec![0.0; query_num];
    let mut cmp_stats: Vec<u32> = vec![0; query_num];
    let mut best_recall = 0.0;

    std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());

    for test_id in 0..l_vec.len() {
        let l_value = l_vec[test_id];

        if l_value < recall_at {
            println!(
                "Ignoring search with L:{} since it's smaller than K:{}",
                l_value, recall_at
            );
            continue;
        }

        let zipped = cmp_stats
            .par_iter_mut()
            .zip(latency_stats.par_iter_mut())
            .zip(query_result_ids[test_id].par_chunks_mut(recall_at as usize))
            .zip(query.par_chunks(query_aligned_dim));

        let start = Instant::now();
        zipped.for_each(|(((cmp, latency), query_result), query_chunk)| {
            let query_start = Instant::now();
            *cmp = index
                .search(query_chunk, recall_at as usize, l_value, query_result)
                .unwrap();

            let query_end = Instant::now();
            let diff = query_end.duration_since(query_start);
            *latency = diff.as_micros() as f32;
        });
        let diff = Instant::now().duration_since(start);

        let mut displayed_qps: f32 = query_num as f32 / diff.as_secs_f32();
        if show_qps_per_thread {
            displayed_qps /= num_threads as f32;
        }

        let mut recalls: Vec<f32> = Vec::new();
        if calc_recall_flag {
            recalls.reserve(recalls_to_print);
            for curr_recall in first_recall..=recall_at {
                recalls.push(search_index_utils::calculate_recall(
                    query_num,
                    gt_ids.as_ref().unwrap(),
                    &gt_dists,
                    gt_dim,
                    &query_result_ids[test_id],
                    recall_at,
                    curr_recall,
                )? as f32);
            }
        }

        latency_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_latency = latency_stats.iter().sum::<f32>() / query_num as f32;
        let avg_cmps = cmp_stats.iter().sum::<u32>() as f32 / query_num as f32;

        let mut stat_str = format!(
            "{: >4}{: >12.2}{: >18.2}{: >20.2}{: >15.2}",
            l_value,
            displayed_qps,
            avg_cmps,
            mean_latency,
            latency_stats[(0.999 * query_num as f32).round() as usize]
        );

        for recall in recalls.iter() {
            stat_str.push_str(&format!("{: >12.2}", recall));
            best_recall = f32::max(best_recall, *recall);
        }

        println!("{}", stat_str);
    }

    println!("Done searching. Now saving results");
    for (test_id, l_value) in l_vec.iter().enumerate() {
        if *l_value < recall_at {
            println!(
                "Ignoring all search with L: {} since it's smaller than K: {}",
                l_value, recall_at
            );
        }

        let cur_result_path = format!("{}_{}_idx_uint32.bin", result_path_prefix, l_value);
        save_bin_u32(
            &cur_result_path,
            query_result_ids[test_id].as_slice(),
            query_num,
            recall_at as usize,
            0,
        )?;
    }

    if best_recall >= fail_if_recall_below {
        Ok(0)
    } else {
        Ok(-1)
    }
}

fn main() -> ANNResult<()> {
    let return_val: i32;
    {
        let mut data_type: String = String::new();
        let mut metric: Option<Metric> = None;
        let mut index_path: String = String::new();
        let mut result_path_prefix: String = String::new();
        let mut query_file: String = String::new();
        let mut truthset_file: String = String::new();
        let mut num_cpus: u32 = num_cpus::get() as u32;
        let mut recall_at: Option<u32> = None;
        let mut print_all_recalls: bool = false;
        let mut l_vec: Vec<u32> = Vec::new();
        let mut show_qps_per_thread: bool = false;
        let mut fail_if_recall_below: f32 = 0.0;

        let args: Vec<String> = env::args().collect();
        let mut iter = args.iter().skip(1).peekable();
        while let Some(arg) = iter.next() {
            let ann_error =
                || ANNError::log_index_config_error(String::from(arg), format!("Missing {}", arg));
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    return Ok(());
                }
                "--data_type" => {
                    data_type = iter.next().ok_or_else(ann_error)?.to_owned();
                }
                "--dist_fn" => {
                    metric = Some(iter.next().ok_or_else(ann_error)?.parse().map_err(|err| {
                        ANNError::log_index_config_error(
                            String::from(arg),
                            format!("ParseError: {}", err),
                        )
                    })?);
                }
                "--index_path_prefix" => {
                    index_path = iter.next().ok_or_else(ann_error)?.to_owned();
                }
                "--result_path" => {
                    result_path_prefix = iter.next().ok_or_else(ann_error)?.to_owned();
                }
                "--query_file" => {
                    query_file = iter.next().ok_or_else(ann_error)?.to_owned();
                }
                "--gt_file" => {
                    truthset_file = iter.next().ok_or_else(ann_error)?.to_owned();
                }
                "--recall_at" | "-K" => {
                    recall_at =
                        Some(iter.next().ok_or_else(ann_error)?.parse().map_err(|err| {
                            ANNError::log_index_config_error(
                                String::from(arg),
                                format!("ParseError: {}", err),
                            )
                        })?);
                }
                "--print_all_recalls" => {
                    print_all_recalls = true;
                }
                "--search_list" | "-L" => {
                    while iter.peek().is_some() && !iter.peek().unwrap().starts_with('-') {
                        l_vec.push(iter.next().ok_or_else(ann_error)?.parse().map_err(|err| {
                            ANNError::log_index_config_error(
                                String::from(arg),
                                format!("ParseError: {}", err),
                            )
                        })?);
                    }
                }
                "--num_threads" => {
                    num_cpus = iter.next().ok_or_else(ann_error)?.parse().map_err(|err| {
                        ANNError::log_index_config_error(
                            String::from(arg),
                            format!("ParseError: {}", err),
                        )
                    })?;
                }
                "--qps_per_thread" => {
                    show_qps_per_thread = true;
                }
                "--fail_if_recall_below" => {
                    fail_if_recall_below =
                        iter.next().ok_or_else(ann_error)?.parse().map_err(|err| {
                            ANNError::log_index_config_error(
                                String::from(arg),
                                format!("ParseError: {}", err),
                            )
                        })?;
                }
                _ => {
                    return Err(ANNError::log_index_error(format!(
                        "Unknown argument: {}",
                        arg
                    )));
                }
            }
        }

        if metric.is_none() {
            return Err(ANNError::log_index_error(String::from("No metric given!")));
        } else if recall_at.is_none() {
            return Err(ANNError::log_index_error(String::from(
                "No recall_at given!",
            )));
        }

        // Seems like float is the only supported data type for FullPrecisionDistance right now,
        // but keep the structure in place here for future data types
        match data_type.as_str() {
            "float" => {
                return_val = search_memory_index::<f32>(
                    metric.unwrap(),
                    &index_path,
                    &result_path_prefix,
                    &query_file,
                    &truthset_file,
                    num_cpus,
                    recall_at.unwrap(),
                    print_all_recalls,
                    &l_vec,
                    show_qps_per_thread,
                    fail_if_recall_below,
                )?;
            }
            "int8" => {
                return_val = search_memory_index::<i8>(
                    metric.unwrap(),
                    &index_path,
                    &result_path_prefix,
                    &query_file,
                    &truthset_file,
                    num_cpus,
                    recall_at.unwrap(),
                    print_all_recalls,
                    &l_vec,
                    show_qps_per_thread,
                    fail_if_recall_below,
                )?;
            }
            "uint8" => {
                return_val = search_memory_index::<u8>(
                    metric.unwrap(),
                    &index_path,
                    &result_path_prefix,
                    &query_file,
                    &truthset_file,
                    num_cpus,
                    recall_at.unwrap(),
                    print_all_recalls,
                    &l_vec,
                    show_qps_per_thread,
                    fail_if_recall_below,
                )?;
            }
            "f16" => {
                return_val = search_memory_index::<Half>(
                    metric.unwrap(),
                    &index_path,
                    &result_path_prefix,
                    &query_file,
                    &truthset_file,
                    num_cpus,
                    recall_at.unwrap(),
                    print_all_recalls,
                    &l_vec,
                    show_qps_per_thread,
                    fail_if_recall_below,
                )?;
            }
            _ => {
                return Err(ANNError::log_index_error(format!(
                    "Unknown data type: {}!",
                    data_type
                )));
            }
        }
    }

    // Rust only allows returning values with this method, but this will immediately terminate the program without running destructors on the
    // stack. To get around this enclose main function logic in a block so that by the time we return here all destructors have been called.
    exit(return_val);
}

fn print_help() {
    println!("Arguments");
    println!("--help, -h                Print information on arguments");
    println!("--data_type               data type <int8/uint8/float> (required)");
    println!("--dist_fn                 distance function <l2/cosine> (required)");
    println!("--index_path_prefix       Path prefix to the index (required)");
    println!("--result_path             Path prefix for saving results of the queries (required)");
    println!("--query_file              Query file in binary format");
    println!("--gt_file                 Ground truth file for the queryset");
    println!("--recall_at, -K           Number of neighbors to be returned");
    println!("--print_all_recalls       Print recalls at all positions, from 1 up to specified recall_at value");
    println!("--search_list             List of L values of search");
    println!("----num_threads, -T       Number of threads used for building index (defaults to num_cpus::get())");
    println!("--qps_per_thread          Print overall QPS divided by the number of threads in the output table");
    println!("--fail_if_recall_below    If set to a value >0 and <100%, program returns -1 if best recall found is below this threshold");
}
