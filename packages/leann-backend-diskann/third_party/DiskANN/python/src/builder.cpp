// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "builder.h"
#include "common.h"
#include "disk_utils.h"
#include "index.h"
#include "parameters.h"

namespace diskannpy
{
template <typename DT>
void build_disk_index(const diskann::Metric metric, const std::string &data_file_path,
                      const std::string &index_prefix_path, const uint32_t complexity, const uint32_t graph_degree,
                      const double final_index_ram_limit, const double indexing_ram_budget, const uint32_t num_threads,
                      const uint32_t pq_disk_bytes, const std::string &codebook_prefix)
{
    std::string params = std::to_string(graph_degree) + " " + std::to_string(complexity) + " " +
                         std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) + " " +
                         std::to_string(num_threads);
    if (pq_disk_bytes > 0)
        params = params + " " + std::to_string(pq_disk_bytes);
    if (!codebook_prefix.empty())
        params = params + " " + codebook_prefix;
    diskann::build_disk_index<DT>(data_file_path.c_str(), index_prefix_path.c_str(), params.c_str(), metric, false,
                                  codebook_prefix);
}

template void build_disk_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                      double, double, uint32_t, uint32_t, const std::string &);

template void build_disk_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        double, double, uint32_t, uint32_t, const std::string &);
template void build_disk_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                       double, double, uint32_t, uint32_t, const std::string &);

template <typename T, typename TagT, typename LabelT>
std::string prepare_filtered_label_map(diskann::Index<T, TagT, LabelT> &index, const std::string &index_output_path,
                                       const std::string &filter_labels_file, const std::string &universal_label)
{
    std::string labels_file_to_use = index_output_path + "_label_formatted.txt";
    std::string mem_labels_int_map_file = index_output_path + "_labels_map.txt";
    convert_labels_string_to_int(filter_labels_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
    if (!universal_label.empty())
    {
        uint32_t unv_label_as_num = 0;
        index.set_universal_label(unv_label_as_num);
    }
    return labels_file_to_use;
}

template std::string prepare_filtered_label_map<float>(diskann::Index<float, uint32_t, uint32_t> &, const std::string &,
                                                       const std::string &, const std::string &);

template std::string prepare_filtered_label_map<int8_t>(diskann::Index<int8_t, uint32_t, uint32_t> &,
                                                        const std::string &, const std::string &, const std::string &);

template std::string prepare_filtered_label_map<uint8_t>(diskann::Index<uint8_t, uint32_t, uint32_t> &,
                                                         const std::string &, const std::string &, const std::string &);

template <typename T, typename TagT, typename LabelT>
void build_memory_index(const diskann::Metric metric, const std::string &vector_bin_path,
                        const std::string &index_output_path, const uint32_t graph_degree, const uint32_t complexity,
                        const float alpha, const uint32_t num_threads, const bool use_pq_build,
                        const size_t num_pq_bytes, const bool use_opq, const bool use_tags,
                        const std::string &filter_labels_file, const std::string &universal_label,
                        const uint32_t filter_complexity)
{
    diskann::IndexWriteParameters index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)
                                                           .with_filter_list_size(filter_complexity)
                                                           .with_alpha(alpha)
                                                           .with_saturate_graph(false)
                                                           .with_num_threads(num_threads)
                                                           .build();
    diskann::IndexSearchParams index_search_params =
        diskann::IndexSearchParams(index_build_params.search_list_size, num_threads);
    size_t data_num, data_dim;
    diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num,
                                          std::make_shared<diskann::IndexWriteParameters>(index_build_params),
                                          std::make_shared<diskann::IndexSearchParams>(index_search_params), 0,
                                          use_tags, use_tags, false, use_pq_build, num_pq_bytes, use_opq);

    if (use_tags)
    {
        const std::string tags_file = index_output_path + ".tags";
        if (!file_exists(tags_file))
        {
            throw std::runtime_error("tags file not found at expected path: " + tags_file);
        }
        TagT *tags_data;
        size_t tag_dims = 1;
        diskann::load_bin(tags_file, tags_data, data_num, tag_dims);
        std::vector<TagT> tags(tags_data, tags_data + data_num);
        if (filter_labels_file.empty())
        {
            index.build(vector_bin_path.c_str(), data_num, tags);
        }
        else
        {
            auto labels_file = prepare_filtered_label_map<T, TagT, LabelT>(index, index_output_path, filter_labels_file,
                                                                           universal_label);
            index.build_filtered_index(vector_bin_path.c_str(), labels_file, data_num, tags);
        }
    }
    else
    {
        if (filter_labels_file.empty())
        {
            index.build(vector_bin_path.c_str(), data_num);
        }
        else
        {
            auto labels_file = prepare_filtered_label_map<T, TagT, LabelT>(index, index_output_path, filter_labels_file,
                                                                           universal_label);
            index.build_filtered_index(vector_bin_path.c_str(), labels_file, data_num);
        }
    }

    index.save(index_output_path.c_str());
}

template void build_memory_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        float, uint32_t, bool, size_t, bool, bool, const std::string &,
                                        const std::string &, uint32_t);

template void build_memory_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                         float, uint32_t, bool, size_t, bool, bool, const std::string &,
                                         const std::string &, uint32_t);

template void build_memory_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                          float, uint32_t, bool, size_t, bool, bool, const std::string &,
                                          const std::string &, uint32_t);

} // namespace diskannpy
