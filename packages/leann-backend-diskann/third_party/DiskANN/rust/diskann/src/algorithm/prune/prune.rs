/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use hashbrown::HashSet;
use vector::{FullPrecisionDistance, Metric};

use crate::common::{ANNError, ANNResult};
use crate::index::InmemIndex;
use crate::model::graph::AdjacencyList;
use crate::model::neighbor::SortedNeighborVector;
use crate::model::scratch::InMemQueryScratch;
use crate::model::Neighbor;

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// A method that occludes a list of neighbors based on some criteria
    #[allow(clippy::too_many_arguments)]
    fn occlude_list(
        &self,
        location: u32,
        pool: &mut SortedNeighborVector,
        alpha: f32,
        degree: u32,
        max_candidate_size: usize,
        result: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
        delete_set_ptr: Option<&HashSet<u32>>,
    ) -> ANNResult<()> {
        if pool.is_empty() {
            return Ok(());
        }

        if !result.is_empty() {
            return Err(ANNError::log_index_error(
                "result is not empty.".to_string(),
            ));
        }

        // Truncate pool at max_candidate_size and initialize scratch spaces
        if pool.len() > max_candidate_size {
            pool.truncate(max_candidate_size);
        }

        let occlude_factor = &mut scratch.occlude_factor;

        // occlude_list can be called with the same scratch more than once by
        // search_for_point_and_add_link through inter_insert.
        occlude_factor.clear();

        // Initialize occlude_factor to pool.len() many 0.0 values for correctness
        occlude_factor.resize(pool.len(), 0.0);

        let mut cur_alpha = 1.0;
        while cur_alpha <= alpha && result.len() < degree as usize {
            for (i, neighbor) in pool.iter().enumerate() {
                if result.len() >= degree as usize {
                    break;
                }
                if occlude_factor[i] > cur_alpha {
                    continue;
                }
                // Set the entry to f32::MAX so that is not considered again
                occlude_factor[i] = f32::MAX;

                // Add the entry to the result if its not been deleted, and doesn't
                // add a self loop
                if delete_set_ptr.map_or(true, |delete_set| !delete_set.contains(&neighbor.id))
                    && neighbor.id != location
                {
                    result.push(neighbor.id);
                }

                // Update occlude factor for points from i+1 to pool.len()
                for (j, neighbor2) in pool.iter().enumerate().skip(i + 1) {
                    if occlude_factor[j] > alpha {
                        continue;
                    }

                    // todo - self.filtered_index
                    let djk = self.get_distance(neighbor2.id, neighbor.id)?;
                    match self.configuration.dist_metric {
                        Metric::L2 | Metric::Cosine => {
                            occlude_factor[j] = if djk == 0.0 {
                                f32::MAX
                            } else {
                                occlude_factor[j].max(neighbor2.distance / djk)
                            };
                        }
                    }
                }
            }

            cur_alpha *= 1.2;
        }

        Ok(())
    }

    /// Prunes the neighbors of a given data point based on some criteria and returns a list of pruned ids.
    ///
    /// # Arguments
    ///
    /// * `location` - The id of the data point whose neighbors are to be pruned.
    /// * `pool` - A vector of neighbors to be pruned, sorted by distance to the query point.
    /// * `pruned_list` - A vector to store the ids of the pruned neighbors.
    /// * `scratch` - A mutable reference to a scratch space for in-memory queries.
    ///
    /// # Panics
    ///
    /// Panics if `pruned_list` contains more than `range` elements after pruning.
    pub fn prune_neighbors(
        &self,
        location: u32,
        pool: &mut Vec<Neighbor>,
        pruned_list: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        self.robust_prune(
            location,
            pool,
            self.configuration.index_write_parameter.max_degree,
            self.configuration.index_write_parameter.max_occlusion_size,
            self.configuration.index_write_parameter.alpha,
            pruned_list,
            scratch,
        )
    }

    /// Prunes the neighbors of a given data point based on some criteria and returns a list of pruned ids.
    ///
    /// # Arguments
    ///
    /// * `location` - The id of the data point whose neighbors are to be pruned.
    /// * `pool` - A vector of neighbors to be pruned, sorted by distance to the query point.
    /// * `range` - The maximum number of neighbors to keep after pruning.
    /// * `max_candidate_size` - The maximum number of candidates to consider for pruning.
    /// * `alpha` - A parameter that controls the occlusion pruning strategy.
    /// * `pruned_list` - A vector to store the ids of the pruned neighbors.
    /// * `scratch` - A mutable reference to a scratch space for in-memory queries.
    ///
    /// # Error
    ///
    /// Return error if `pruned_list` contains more than `range` elements after pruning.
    #[allow(clippy::too_many_arguments)]
    fn robust_prune(
        &self,
        location: u32,
        pool: &mut Vec<Neighbor>,
        range: u32,
        max_candidate_size: u32,
        alpha: f32,
        pruned_list: &mut AdjacencyList,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        if pool.is_empty() {
            // if the pool is empty, behave like a noop
            pruned_list.clear();
            return Ok(());
        }

        // If using _pq_build, over-write the PQ distances with actual distances
        // todo : pq_dist

        // sort the pool based on distance to query and prune it with occlude_list
        let mut pool = SortedNeighborVector::new(pool);
        pruned_list.clear();

        self.occlude_list(
            location,
            &mut pool,
            alpha,
            range,
            max_candidate_size as usize,
            pruned_list,
            scratch,
            Option::None,
        )?;

        if pruned_list.len() > range as usize {
            return Err(ANNError::log_index_error(format!(
                "pruned_list's len {} is over range {}.",
                pruned_list.len(),
                range
            )));
        }

        if self.configuration.index_write_parameter.saturate_graph && alpha > 1.0f32 {
            for neighbor in pool.iter() {
                if pruned_list.len() >= (range as usize) {
                    break;
                }
                if !pruned_list.contains(&neighbor.id) && neighbor.id != location {
                    pruned_list.push(neighbor.id);
                }
            }
        }

        Ok(())
    }

    /// A method that inserts a point n into the graph of its neighbors and their neighbors,
    /// pruning the graph if necessary to keep it within the specified range
    /// * `n` - The index of the new point
    /// * `pruned_list` is a vector of the neighbors of n that have been pruned by a previous step
    /// * `range` is the target number of neighbors for each point
    /// * `scratch` is a mutable reference to a scratch space that can be reused for intermediate computations
    pub fn inter_insert(
        &self,
        n: u32,
        pruned_list: &Vec<u32>,
        range: u32,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> ANNResult<()> {
        // Borrow the pruned_list as a source pool of neighbors
        let src_pool = pruned_list;

        if src_pool.is_empty() {
            return Err(ANNError::log_index_error("src_pool is empty.".to_string()));
        }

        for &vertex_id in src_pool {
            // vertex is the index of a neighbor of n
            // Assert that vertex is within the valid range of points
            if (vertex_id as usize)
                >= self.configuration.max_points + self.configuration.num_frozen_pts
            {
                return Err(ANNError::log_index_error(format!(
                    "vertex_id {} is out of valid range of points {}",
                    vertex_id,
                    self.configuration.max_points + self.configuration.num_frozen_pts,
                )));
            }

            let neighbors = self.add_to_neighbors(vertex_id, n, range)?;

            if let Some(copy_of_neighbors) = neighbors {
                // Pruning is needed, create a dummy set and a dummy vector to store the unique neighbors of vertex_id
                let mut dummy_pool = self.get_unique_neighbors(&copy_of_neighbors, vertex_id)?;

                // Create a new vector to store the pruned neighbors of vertex_id
                let mut new_out_neighbors =
                    AdjacencyList::for_range(self.configuration.write_range());
                // Prune the neighbors of vertex_id using a helper method
                self.prune_neighbors(vertex_id, &mut dummy_pool, &mut new_out_neighbors, scratch)?;

                self.set_neighbors(vertex_id, new_out_neighbors)?;
            }
        }

        Ok(())
    }

    /// Adds a node to the list of neighbors for the given node.
    ///
    /// # Arguments
    ///
    /// * `vertex_id` - The ID of the node to add the neighbor to.
    /// * `node_id` - The ID of the node to add.
    /// * `range` - The range of the graph.
    ///
    /// # Return
    ///
    /// Returns `None` if the node is already in the list of neighbors, or a `Vec` containing the updated list of neighbors if the list of neighbors is full.
    fn add_to_neighbors(
        &self,
        vertex_id: u32,
        node_id: u32,
        range: u32,
    ) -> ANNResult<Option<Vec<u32>>> {
        // vertex contains a vector of the neighbors of vertex_id
        let mut vertex_guard = self.final_graph.write_vertex_and_neighbors(vertex_id)?;

        Ok(vertex_guard.add_to_neighbors(node_id, range))
    }

    fn set_neighbors(&self, vertex_id: u32, new_out_neighbors: AdjacencyList) -> ANNResult<()> {
        // vertex contains a vector of the neighbors of vertex_id
        let mut vertex_guard = self.final_graph.write_vertex_and_neighbors(vertex_id)?;

        vertex_guard.set_neighbors(new_out_neighbors);
        Ok(())
    }
}

