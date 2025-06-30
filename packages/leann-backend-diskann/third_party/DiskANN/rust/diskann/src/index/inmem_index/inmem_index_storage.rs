/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use vector::FullPrecisionDistance;

use crate::common::{ANNError, ANNResult};
use crate::model::graph::AdjacencyList;
use crate::model::InMemoryGraph;
use crate::utils::{file_exists, save_data_in_base_dimensions};

use super::InmemIndex;

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn load_graph(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<usize> {
        // let file_offset = 0; // will need this for single file format support

        let mut in_file = BufReader::new(File::open(Path::new(filename))?);
        // in_file.seek(SeekFrom::Start(file_offset as u64))?;

        let expected_file_size: usize = in_file.read_u64::<LittleEndian>()? as usize;
        self.max_observed_degree = in_file.read_u32::<LittleEndian>()?;
        self.start = in_file.read_u32::<LittleEndian>()?;
        let file_frozen_pts: usize = in_file.read_u64::<LittleEndian>()? as usize;

        let vamana_metadata_size = 24;

        println!("From graph header, expected_file_size: {}, max_observed_degree: {}, start: {}, file_frozen_pts: {}",
            expected_file_size, self.max_observed_degree, self.start, file_frozen_pts);

        if file_frozen_pts != self.configuration.num_frozen_pts {
            if file_frozen_pts == 1 {
                return Err(ANNError::log_index_config_error(
                    "num_frozen_pts".to_string(), 
                    "ERROR: When loading index, detected dynamic index, but constructor asks for static index. Exitting.".to_string())
                );
            } else {
                return Err(ANNError::log_index_config_error(
                    "num_frozen_pts".to_string(), 
                    "ERROR: When loading index, detected static index, but constructor asks for dynamic index. Exitting.".to_string())
                );
            }
        }

        println!("Loading vamana graph {}...", filename);

        let expected_max_points = expected_num_points - file_frozen_pts;

        // If user provides more points than max_points
        // resize the _final_graph to the larger size.
        if self.configuration.max_points < expected_max_points {
            println!("Number of points in data: {} is greater than max_points: {} Setting max points to: {}", expected_max_points, self.configuration.max_points, expected_max_points);

            self.configuration.max_points = expected_max_points;
            self.final_graph = InMemoryGraph::new(
                self.configuration.max_points + self.configuration.num_frozen_pts,
                self.configuration.index_write_parameter.max_degree,
            );
        }

        let mut bytes_read = vamana_metadata_size;
        let mut num_edges = 0;
        let mut nodes_read = 0;
        let mut max_observed_degree = 0;

        while bytes_read != expected_file_size {
            let num_nbrs = in_file.read_u32::<LittleEndian>()?;
            max_observed_degree = if num_nbrs > max_observed_degree {
                num_nbrs
            } else {
                max_observed_degree
            };

            if num_nbrs == 0 {
                return Err(ANNError::log_index_error(format!(
                    "ERROR: Point found with no out-neighbors, point# {}",
                    nodes_read
                )));
            }

            num_edges += num_nbrs;
            nodes_read += 1;
            let mut tmp: Vec<u32> = Vec::with_capacity(num_nbrs as usize);
            for _ in 0..num_nbrs {
                tmp.push(in_file.read_u32::<LittleEndian>()?);
            }

            self.final_graph
                .write_vertex_and_neighbors(nodes_read - 1)?
                .set_neighbors(AdjacencyList::from(tmp));
            bytes_read += 4 * (num_nbrs as usize + 1);
        }

        println!(
            "Done. Index has {} nodes and {} out-edges, _start is set to {}",
            nodes_read, num_edges, self.start
        );

        self.max_observed_degree = max_observed_degree;
        Ok(nodes_read as usize)
    }

    /// Save the graph index on a file as an adjacency list.
    /// For each point, first store the number of neighbors,
    /// and then the neighbor list (each as 4 byte u32)
    pub fn save_graph(&mut self, graph_file: &str) -> ANNResult<u64> {
        let file: File = File::create(graph_file)?;
        let mut out = BufWriter::new(file);

        let file_offset: u64 = 0;
        out.seek(SeekFrom::Start(file_offset))?;
        let mut index_size: u64 = 24;
        let mut max_degree: u32 = 0;
        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&self.max_observed_degree.to_le_bytes())?;
        out.write_all(&self.start.to_le_bytes())?;
        out.write_all(&(self.configuration.num_frozen_pts as u64).to_le_bytes())?;

        // At this point, either nd == max_points or any frozen points have
        // been temporarily moved to nd, so nd + num_frozen_points is the valid
        // location limit
        for i in 0..self.num_active_pts + self.configuration.num_frozen_pts {
            let idx = i as u32;
            let gk: u32 = self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32;
            out.write_all(&gk.to_le_bytes())?;
            for neighbor in self
                .final_graph
                .read_vertex_and_neighbors(idx)?
                .get_neighbors()
                .iter()
            {
                out.write_all(&neighbor.to_le_bytes())?;
            }
            max_degree =
                if self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32 > max_degree {
                    self.final_graph.read_vertex_and_neighbors(idx)?.size() as u32
                } else {
                    max_degree
                };
            index_size += (std::mem::size_of::<u32>() * (gk as usize + 1)) as u64;
        }
        out.seek(SeekFrom::Start(file_offset))?;
        out.write_all(&index_size.to_le_bytes())?;
        out.write_all(&max_degree.to_le_bytes())?;
        out.flush()?;
        Ok(index_size)
    }

    /// Save the data on a file.
    pub fn save_data(&mut self, data_file: &str) -> ANNResult<usize> {
        // Note: at this point, either _nd == _max_points or any frozen points have
        // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
        // location limit.
        Ok(save_data_in_base_dimensions(
            data_file,
            &mut self.dataset.data,
            self.num_active_pts + self.configuration.num_frozen_pts,
            self.configuration.dim,
            self.configuration.aligned_dim,
            0,
        )?)
    }

    /// Save the delete list to a file only if the delete list length is not zero.
    pub fn save_delete_list(&mut self, delete_list_file: &str) -> ANNResult<usize> {
        let mut delete_file_size = 0;
        if let Ok(delete_set) = self.delete_set.read() {
            let delete_set_len = delete_set.len() as u32;

            if delete_set_len != 0 {
                let file: File = File::create(delete_list_file)?;
                let mut writer = BufWriter::new(file);

                // Write the length of the set.
                writer.write_all(&delete_set_len.to_le_bytes())?;
                delete_file_size += std::mem::size_of::<u32>();

                // Write the elements of the set.
                for &item in delete_set.iter() {
                    writer.write_all(&item.to_be_bytes())?;
                    delete_file_size += std::mem::size_of::<u32>();
                }

                writer.flush()?;
            }
        } else {
            return Err(ANNError::log_lock_poison_error(
                "Poisoned lock on delete set. Can't save deleted list.".to_string(),
            ));
        }

        Ok(delete_file_size)
    }

    // load the deleted list from the delete file if it exists.
    pub fn load_delete_list(&mut self, delete_list_file: &str) -> ANNResult<usize> {
        let mut len = 0;

        if file_exists(delete_list_file) {
            let file = File::open(delete_list_file)?;
            let mut reader = BufReader::new(file);

            len = reader.read_u32::<LittleEndian>()? as usize;

            if let Ok(mut delete_set) = self.delete_set.write() {
                for _ in 0..len {
                    let item = reader.read_u32::<LittleEndian>()?;
                    delete_set.insert(item);
                }
            } else {
                return Err(ANNError::log_lock_poison_error(
                    "Poisoned lock on delete set. Can't load deleted list.".to_string(),
                ));
            }
        }

        Ok(len)
    }
}

#[cfg(test)]
mod index_test {
    use std::fs;

    use vector::Metric;

    use super::*;
    use crate::{
        index::ANNInmemIndex,
        model::{
            configuration::index_write_parameters::IndexWriteParametersBuilder, vertex::DIM_128,
            IndexConfiguration,
        },
        utils::{load_metadata_from_file, round_up},
    };

    const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
    const R: u32 = 4;
    const L: u32 = 50;
    const ALPHA: f32 = 1.2;

    #[cfg_attr(not(coverage), test)]
    fn save_graph_test() {
        let parameters = IndexWriteParametersBuilder::new(50, 4)
            .with_alpha(1.2)
            .build();
        let config =
            IndexConfiguration::new(Metric::L2, 10, 16, 16, false, 0, false, 8, 1f32, parameters);
        let mut index = InmemIndex::<f32, 3>::new(config).unwrap();
        let final_graph = InMemoryGraph::new(10, 3);
        let num_active_pts = 2_usize;
        index.final_graph = final_graph;
        index.num_active_pts = num_active_pts;
        let graph_file = "test_save_graph_data.bin";
        let result = index.save_graph(graph_file);
        assert!(result.is_ok());

        fs::remove_file(graph_file).expect("Failed to delete file");
    }

    #[test]
    fn save_data_test() {
        let (data_num, dim) = load_metadata_from_file(TEST_DATA_FILE).unwrap();

        let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
            .with_alpha(ALPHA)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            dim,
            round_up(dim as u64, 16_u64) as usize,
            data_num,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
        );
        let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config).unwrap();

        index.build(TEST_DATA_FILE, data_num).unwrap();

        let data_file = "test.data";
        let result = index.save_data(data_file);
        assert_eq!(
            result.unwrap(),
            2 * std::mem::size_of::<u32>()
                + (index.num_active_pts + index.configuration.num_frozen_pts)
                    * index.configuration.dim
                    * (std::mem::size_of::<f32>())
        );
        fs::remove_file(data_file).expect("Failed to delete file");
    }
}
