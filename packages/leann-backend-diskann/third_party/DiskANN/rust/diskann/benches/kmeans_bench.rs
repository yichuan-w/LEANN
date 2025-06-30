/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use criterion::{criterion_group, criterion_main, Criterion};
use diskann::utils::k_means_clustering;
use rand::Rng;

const NUM_POINTS: usize = 10000;
const DIM: usize = 100;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

fn benchmark_kmeans_rust(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..NUM_POINTS * DIM)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let centers: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];

    let mut group = c.benchmark_group("kmeans-computation");
    group.sample_size(500);

    group.bench_function("K-Means Rust run", |f| {
        f.iter(|| {
            // let mut centers_copy = centers.clone();
            let data_copy = data.clone();
            let mut centers_copy = centers.clone();
            k_means_clustering(
                &data_copy,
                NUM_POINTS,
                DIM,
                &mut centers_copy,
                NUM_CENTERS,
                MAX_KMEANS_REPS,
            )
        })
    });
}

fn benchmark_kmeans_c(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..NUM_POINTS * DIM)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let centers: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];

    let mut group = c.benchmark_group("kmeans-computation");
    group.sample_size(500);

    group.bench_function("K-Means C++ Run", |f| {
        f.iter(|| {
            let data_copy = data.clone();
            let mut centers_copy = centers.clone();
            let _ = k_means_clustering(
                data_copy.as_slice(),
                NUM_POINTS,
                DIM,
                centers_copy.as_mut_slice(),
                NUM_CENTERS,
                MAX_KMEANS_REPS,
            );
        })
    });
}

criterion_group!(benches, benchmark_kmeans_rust, benchmark_kmeans_c);

criterion_main!(benches);

