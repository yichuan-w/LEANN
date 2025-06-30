/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use diskann::model::{Neighbor, NeighborPriorityQueue};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn benchmark_priority_queue_insert(c: &mut Criterion) {
    let vec = generate_random_floats();
    let mut group = c.benchmark_group("neighborqueue-insert");
    group.measurement_time(Duration::from_secs(3)).sample_size(500);

    let mut queue = NeighborPriorityQueue::with_capacity(64_usize);
    group.bench_function("Neighbor Priority Queue Insert", |f| {
        f.iter(|| {
            queue.clear();
            for n in vec.iter() {
                queue.insert(*n);
            }

            black_box(&1)
        });
    });
}

fn generate_random_floats() -> Vec<Neighbor> {
    let seed: [u8; 32] = [73; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Uniform::new(0.0, 1.0);
    let mut random_floats = Vec::with_capacity(100);

    for i in 0..100 {
        let random_float = range.sample(&mut rng) as f32;
        let n = Neighbor::new(i, random_float);
        random_floats.push(n);
    }

    random_floats
}

criterion_group!(benches, benchmark_priority_queue_insert);
criterion_main!(benches);

