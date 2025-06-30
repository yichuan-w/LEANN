/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Index write parameters.

/// Default parameter values.
pub mod default_param_vals {
    /// Default value of alpha.
    pub const ALPHA: f32 = 1.2;

    /// Default value of number of threads.
    pub const NUM_THREADS: u32 = 0;

    /// Default value of number of rounds.
    pub const NUM_ROUNDS: u32 = 2;

    /// Default value of max occlusion size.
    pub const MAX_OCCLUSION_SIZE: u32 = 750;

    /// Default value of filter list size.
    pub const FILTER_LIST_SIZE: u32 = 0;

    /// Default value of number of frozen points.
    pub const NUM_FROZEN_POINTS: u32 = 0;
    
    /// Default value of max degree.
    pub const MAX_DEGREE: u32 = 64;

    /// Default value of build list size.
    pub const BUILD_LIST_SIZE: u32 = 100;

    /// Default value of saturate graph.
    pub const SATURATE_GRAPH: bool = false;

    /// Default value of search list size.
    pub const SEARCH_LIST_SIZE: u32 = 100;
}

/// Index write parameters.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IndexWriteParameters {
    /// Search list size - L.
    pub search_list_size: u32,

    /// Max degree - R.
    pub max_degree: u32,

    /// Saturate graph.
    pub saturate_graph: bool,

    /// Max occlusion size - C.
    pub max_occlusion_size: u32,

    /// Alpha.
    pub alpha: f32,

    /// Number of rounds.
    pub num_rounds: u32,

    /// Number of threads.
    pub num_threads: u32,
    
    /// Number of frozen points.
    pub num_frozen_points: u32,
}

impl Default for IndexWriteParameters {
    /// Create IndexWriteParameters with default values
    fn default() -> Self {
        Self {
            search_list_size: default_param_vals::SEARCH_LIST_SIZE,
            max_degree: default_param_vals::MAX_DEGREE,
            saturate_graph: default_param_vals::SATURATE_GRAPH,
            max_occlusion_size: default_param_vals::MAX_OCCLUSION_SIZE,
            alpha: default_param_vals::ALPHA,
            num_rounds: default_param_vals::NUM_ROUNDS,
            num_threads: default_param_vals::NUM_THREADS,
            num_frozen_points: default_param_vals::NUM_FROZEN_POINTS
        }
    }
}

/// The builder for IndexWriteParameters.
#[derive(Debug)]
pub struct IndexWriteParametersBuilder {
    search_list_size: u32,
    max_degree: u32,
    max_occlusion_size: Option<u32>,
    saturate_graph: Option<bool>,
    alpha: Option<f32>,
    num_rounds: Option<u32>,
    num_threads: Option<u32>,
    // filter_list_size: Option<u32>,
    num_frozen_points: Option<u32>,
}

impl IndexWriteParametersBuilder {
    /// Initialize IndexWriteParametersBuilder
    pub fn new(search_list_size: u32, max_degree: u32) -> Self {
        Self {
            search_list_size,
            max_degree,
            max_occlusion_size: None,
            saturate_graph: None,
            alpha: None,
            num_rounds: None,
            num_threads: None,
            // filter_list_size: None,
            num_frozen_points: None,
        }
    }

    /// Set max occlusion size.
    pub fn with_max_occlusion_size(mut self, max_occlusion_size: u32) -> Self {
        self.max_occlusion_size = Some(max_occlusion_size);
        self
    }

    /// Set saturate graph.
    pub fn with_saturate_graph(mut self, saturate_graph: bool) -> Self {
        self.saturate_graph = Some(saturate_graph);
        self
    }

    /// Set alpha.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Set number of rounds.
    pub fn with_num_rounds(mut self, num_rounds: u32) -> Self {
        self.num_rounds = Some(num_rounds);
        self
    }

    /// Set number of threads.
    pub fn with_num_threads(mut self, num_threads: u32) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /*
    pub fn with_filter_list_size(mut self, filter_list_size: u32) -> Self {
        self.filter_list_size = Some(filter_list_size);
        self
    }
    */

    /// Set number of frozen points.
    pub fn with_num_frozen_points(mut self, num_frozen_points: u32) -> Self {
        self.num_frozen_points = Some(num_frozen_points);
        self
    }

    /// Build IndexWriteParameters from IndexWriteParametersBuilder.
    pub fn build(self) -> IndexWriteParameters {
        IndexWriteParameters {
            search_list_size: self.search_list_size,
            max_degree: self.max_degree,
            saturate_graph: self.saturate_graph.unwrap_or(default_param_vals::SATURATE_GRAPH),
            max_occlusion_size: self.max_occlusion_size.unwrap_or(default_param_vals::MAX_OCCLUSION_SIZE),
            alpha: self.alpha.unwrap_or(default_param_vals::ALPHA),
            num_rounds: self.num_rounds.unwrap_or(default_param_vals::NUM_ROUNDS),
            num_threads: self.num_threads.unwrap_or(default_param_vals::NUM_THREADS),
            // filter_list_size: self.filter_list_size.unwrap_or(default_param_vals::FILTER_LIST_SIZE),
            num_frozen_points: self.num_frozen_points.unwrap_or(default_param_vals::NUM_FROZEN_POINTS),
        }
    }
}

/// Construct IndexWriteParametersBuilder from IndexWriteParameters.
impl From<IndexWriteParameters> for IndexWriteParametersBuilder {
    fn from(param: IndexWriteParameters) -> Self {
        Self {
            search_list_size: param.search_list_size,
            max_degree: param.max_degree,
            max_occlusion_size: Some(param.max_occlusion_size),
            saturate_graph: Some(param.saturate_graph),
            alpha: Some(param.alpha),
            num_rounds: Some(param.num_rounds),
            num_threads: Some(param.num_threads),
            // filter_list_size: Some(param.filter_list_size),
            num_frozen_points: Some(param.num_frozen_points),
        }
    }
}

#[cfg(test)]
mod parameters_test {
    use crate::model::configuration::index_write_parameters::*;

    #[test]
    fn test_default_index_params() {
        let wp1 = IndexWriteParameters::default();
        assert_eq!(wp1.search_list_size, default_param_vals::SEARCH_LIST_SIZE);
        assert_eq!(wp1.max_degree, default_param_vals::MAX_DEGREE);
        assert_eq!(wp1.saturate_graph, default_param_vals::SATURATE_GRAPH);
        assert_eq!(wp1.max_occlusion_size, default_param_vals::MAX_OCCLUSION_SIZE);
        assert_eq!(wp1.alpha, default_param_vals::ALPHA);
        assert_eq!(wp1.num_rounds, default_param_vals::NUM_ROUNDS);
        assert_eq!(wp1.num_threads, default_param_vals::NUM_THREADS);
        assert_eq!(wp1.num_frozen_points, default_param_vals::NUM_FROZEN_POINTS);
    }

    #[test]
    fn test_index_write_parameters_builder() {
        // default value
        let wp1 = IndexWriteParametersBuilder::new(10, 20).build();
        assert_eq!(wp1.search_list_size, 10);
        assert_eq!(wp1.max_degree, 20);
        assert_eq!(wp1.saturate_graph, default_param_vals::SATURATE_GRAPH);
        assert_eq!(wp1.max_occlusion_size, default_param_vals::MAX_OCCLUSION_SIZE);
        assert_eq!(wp1.alpha, default_param_vals::ALPHA);
        assert_eq!(wp1.num_rounds, default_param_vals::NUM_ROUNDS);
        assert_eq!(wp1.num_threads, default_param_vals::NUM_THREADS);
        assert_eq!(wp1.num_frozen_points, default_param_vals::NUM_FROZEN_POINTS);
    
        // build with custom values
        let wp2 = IndexWriteParametersBuilder::new(10, 20)
            .with_max_occlusion_size(30)
            .with_saturate_graph(true)
            .with_alpha(0.5)
            .with_num_rounds(40)
            .with_num_threads(50)
            .with_num_frozen_points(60)
            .build();
        assert_eq!(wp2.search_list_size, 10);
        assert_eq!(wp2.max_degree, 20);
        assert!(wp2.saturate_graph);
        assert_eq!(wp2.max_occlusion_size, 30);
        assert_eq!(wp2.alpha, 0.5);
        assert_eq!(wp2.num_rounds, 40);
        assert_eq!(wp2.num_threads, 50);
        assert_eq!(wp2.num_frozen_points, 60);
    
        // test from
        let wp3 = IndexWriteParametersBuilder::from(wp2).build();
        assert_eq!(wp3, wp2);
    }
}

