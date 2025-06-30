/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::{thread_rng, Rng};
use vector::{FullPrecisionDistance, Metric};

// make sure the vector is 256-bit (32 bytes) aligned required by _mm256_load_ps
#[repr(C, align(32))]
struct Vector32ByteAligned {
    v: [f32; 256],
}

fn benchmark_l2_distance_float_rust(c: &mut Criterion) {
    let (a, b) = prepare_random_aligned_vectors();
    let mut group = c.benchmark_group("avx-computation");
    group.sample_size(5000);

    group.bench_function("AVX Rust run", |f| {
        f.iter(|| {
            black_box(<[f32; 256]>::distance_compare(
                black_box(&a.v),
                black_box(&b.v),
                Metric::L2,
            ))
        })
    });
}

// make sure the vector is 256-bit (32 bytes) aligned required by _mm256_load_ps
fn prepare_random_aligned_vectors() -> (Box<Vector32ByteAligned>, Box<Vector32ByteAligned>) {
    let a = Box::new(Vector32ByteAligned {
        v: [(); 256].map(|_| thread_rng().gen_range(0.0..100.0)),
    });

    let b = Box::new(Vector32ByteAligned {
        v: [(); 256].map(|_| thread_rng().gen_range(0.0..100.0)),
    });

    (a, b)
}

criterion_group!(benches, benchmark_l2_distance_float_rust,);
criterion_main!(benches);

