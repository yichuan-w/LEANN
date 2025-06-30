/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! In-memory Dataset

use rayon::prelude::*;
use std::mem;
use vector::{FullPrecisionDistance, Metric};

use crate::common::{ANNError, ANNResult, AlignedBoxWithSlice};
use crate::model::Vertex;
use crate::utils::copy_aligned_data_from_file;

/// Dataset of all in-memory FP points
#[derive(Debug)]
pub struct InmemDataset<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// All in-memory points
    pub data: AlignedBoxWithSlice<T>,

    /// Number of points we anticipate to have
    pub num_points: usize,

    /// Number of active points i.e. existing in the graph
    pub num_active_pts: usize,

    /// Capacity of the dataset
    pub capacity: usize,
}

impl<'a, T, const N: usize> InmemDataset<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create the dataset with size num_points and growth factor.
    /// growth factor=1 means no growth (provision 100% space of num_points)
    /// growth factor=1.2 means provision 120% space of num_points (20% extra space)
    pub fn new(num_points: usize, index_growth_factor: f32) -> ANNResult<Self> {
        let capacity = (((num_points * N) as f32) * index_growth_factor) as usize;

        Ok(Self {
            data: AlignedBoxWithSlice::new(capacity, mem::size_of::<T>() * 16)?,
            num_points,
            num_active_pts: num_points,
            capacity,
        })
    }

    /// get immutable data slice
    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    /// Build the dataset from file
    pub fn build_from_file(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()> {
        println!(
            "Loading {} vectors from file {} into dataset...",
            num_points_to_load, filename
        );
        self.num_active_pts = num_points_to_load;

        copy_aligned_data_from_file(filename, self.into_dto(), 0)?;

        println!("Dataset loaded.");
        Ok(())
    }

    /// Append the dataset from file
    pub fn append_from_file(
        &mut self,
        filename: &str,
        num_points_to_append: usize,
    ) -> ANNResult<()> {
        println!(
            "Appending {} vectors from file {} into dataset...",
            num_points_to_append, filename
        );
        if self.num_points + num_points_to_append > self.capacity {
            return Err(ANNError::log_index_error(format!(
                "Cannot append {} points to dataset of capacity {}",
                num_points_to_append, self.capacity
            )));
        }

        let pts_offset = self.num_active_pts;
        copy_aligned_data_from_file(filename, self.into_dto(), pts_offset)?;

        self.num_active_pts += num_points_to_append;
        self.num_points += num_points_to_append;

        println!("Dataset appended.");
        Ok(())
    }

    /// Get vertex by id
    pub fn get_vertex(&'a self, id: u32) -> ANNResult<Vertex<'a, T, N>> {
        let start = id as usize * N;
        let end = start + N;

        if end <= self.data.len() {
            let val = <&[T; N]>::try_from(&self.data[start..end]).map_err(|err| {
                ANNError::log_index_error(format!("Failed to get vertex {}, err={}", id, err))
            })?;
            Ok(Vertex::new(val, id))
        } else {
            Err(ANNError::log_index_error(format!(
                "Invalid vertex id {}.",
                id
            )))
        }
    }

    /// Get full precision distance between two nodes
    pub fn get_distance(&self, id1: u32, id2: u32, metric: Metric) -> ANNResult<f32> {
        let vertex1 = self.get_vertex(id1)?;
        let vertex2 = self.get_vertex(id2)?;

        Ok(vertex1.compare(&vertex2, metric))
    }

    /// find out the medoid, the vertex in the dataset that is closest to the centroid
    pub fn calculate_medoid_point_id(&self) -> ANNResult<u32> {
        Ok(self.find_nearest_point_id(self.calculate_centroid_point()?))
    }

    /// calculate centroid, average of all vertices in the dataset
    fn calculate_centroid_point(&self) -> ANNResult<[f32; N]> {
        // Allocate and initialize the centroid vector
        let mut center: [f32; N] = [0.0; N];

        // Sum the data points' components
        for i in 0..self.num_active_pts {
            let vertex = self.get_vertex(i as u32)?;
            let vertex_slice = vertex.vector();
            for j in 0..N {
                center[j] += vertex_slice[j].into();
            }
        }

        // Divide by the number of points to calculate the centroid
        let capacity = self.num_active_pts as f32;
        for item in center.iter_mut().take(N) {
            *item /= capacity;
        }

        Ok(center)
    }

    /// find out the vertex closest to the given point
    fn find_nearest_point_id(&self, point: [f32; N]) -> u32 {
        // compute all to one distance
        let mut distances = vec![0f32; self.num_active_pts];
        let slice = &self.data[..];
        distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
            let start = i * N;
            for j in 0..N {
                let diff: f32 = (point.as_slice()[j] - slice[start + j].into())
                    * (point.as_slice()[j] - slice[start + j].into());
                *dist += diff;
            }
        });

        let mut min_idx = 0;
        let mut min_dist = f32::MAX;
        for (i, distance) in distances.iter().enumerate().take(self.num_active_pts) {
            if *distance < min_dist {
                min_idx = i;
                min_dist = *distance;
            }
        }
        min_idx as u32
    }

    /// Prefetch vertex data in the memory hierarchy
    /// NOTE: good efficiency when total_vec_size is integral multiple of 64
    #[inline]
    pub fn prefetch_vector(&self, id: u32) {
        let start = id as usize * N;
        let end = start + N;

        if end <= self.data.len() {
            let vec = &self.data[start..end];
            vector::prefetch_vector(vec);
        }
    }

    /// Convert into dto object
    pub fn into_dto(&mut self) -> DatasetDto<T> {
        DatasetDto { 
            data: &mut self.data,
            rounded_dim: N,
        }
    }
}

/// Dataset dto used for other layer, such as storage
/// N is the aligned dimension
#[derive(Debug)]
pub struct DatasetDto<'a, T> {
    /// data slice borrow from dataset
    pub data: &'a mut [T],

    /// rounded dimension
    pub rounded_dim: usize,
}

#[cfg(test)]
mod dataset_test {
    use std::fs;

    use super::*;
    use crate::model::vertex::DIM_128;

    #[test]
    fn get_vertex_within_range() {
        let num_points = 1_000_000;
        let id = 999_999;
        let dataset = InmemDataset::<f32, DIM_128>::new(num_points, 1f32).unwrap();

        let vertex = dataset.get_vertex(999_999).unwrap();

        assert_eq!(vertex.vertex_id(), id);
        assert_eq!(vertex.vector().len(), DIM_128);
        assert_eq!(vertex.vector().as_ptr(), unsafe {
            dataset.data.as_ptr().add((id as usize) * DIM_128)
        });
    }

    #[test]
    fn get_vertex_out_of_range() {
        let num_points = 1_000_000;
        let invalid_id = 1_000_000;
        let dataset = InmemDataset::<f32, DIM_128>::new(num_points, 1f32).unwrap();

        if dataset.get_vertex(invalid_id).is_ok() {
            panic!("id ({}) should be out of range", invalid_id)
        };
    }

    #[test]
    fn load_data_test() {
        let file_name = "dataset_test_load_data_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let mut dataset = InmemDataset::<f32, 8>::new(2, 1f32).unwrap();

        match copy_aligned_data_from_file(
            file_name,
            dataset.into_dto(),
            0,
        ) {
            Ok((npts, dim)) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                assert!(npts == 2);
                assert!(dim == 8);
                assert!(dataset.data.len() == 16);

                let first_vertex = dataset.get_vertex(0).unwrap();
                let second_vertex = dataset.get_vertex(1).unwrap();

                assert!(*first_vertex.vector() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
                assert!(*second_vertex.vector() == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
            }
            Err(e) => {
                fs::remove_file(file_name).expect("Failed to delete file");
                panic!("{}", e)
            }
        }
    }
}

