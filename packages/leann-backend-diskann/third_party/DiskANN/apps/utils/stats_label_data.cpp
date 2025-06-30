// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <string.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <set>
#include <boost/program_options.hpp>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#else
#include <Windows.h>
#endif
namespace po = boost::program_options;

void stats_analysis(const std::string labels_file, std::string univeral_label, uint32_t density = 10)
{
    std::string token, line;
    std::ifstream labels_stream(labels_file);
    std::unordered_map<std::string, uint32_t> label_counts;
    std::string label_with_max_points;
    uint32_t max_points = 0;
    long long sum = 0;
    long long point_cnt = 0;
    float avg_labels_per_pt, mean_label_size;

    std::vector<uint32_t> labels_per_point;
    uint32_t dense_pts = 0;
    if (labels_stream.is_open())
    {
        while (getline(labels_stream, line))
        {
            point_cnt++;
            std::stringstream iss(line);
            uint32_t lbl_cnt = 0;
            while (getline(iss, token, ','))
            {
                lbl_cnt++;
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                if (label_counts.find(token) == label_counts.end())
                    label_counts[token] = 0;
                label_counts[token]++;
            }
            if (lbl_cnt >= density)
            {
                dense_pts++;
            }
            labels_per_point.emplace_back(lbl_cnt);
        }
    }

    std::cout << "fraction of dense points with >= " << density
              << " labels = " << (float)dense_pts / (float)labels_per_point.size() << std::endl;
    std::sort(labels_per_point.begin(), labels_per_point.end());

    std::vector<std::pair<std::string, uint32_t>> label_count_vec;

    for (auto it = label_counts.begin(); it != label_counts.end(); it++)
    {
        auto &lbl = *it;
        label_count_vec.emplace_back(std::make_pair(lbl.first, lbl.second));
        if (lbl.second > max_points)
        {
            max_points = lbl.second;
            label_with_max_points = lbl.first;
        }
        sum += lbl.second;
    }

    sort(label_count_vec.begin(), label_count_vec.end(),
         [](const std::pair<std::string, uint32_t> &lhs, const std::pair<std::string, uint32_t> &rhs) {
             return lhs.second < rhs.second;
         });

    for (float p = 0; p < 1; p += 0.05)
    {
        std::cout << "Percentile " << (100 * p) << "\t" << label_count_vec[(size_t)(p * label_count_vec.size())].first
                  << " with count=" << label_count_vec[(size_t)(p * label_count_vec.size())].second << std::endl;
    }

    std::cout << "Most common label "
              << "\t" << label_count_vec[label_count_vec.size() - 1].first
              << " with count=" << label_count_vec[label_count_vec.size() - 1].second << std::endl;
    if (label_count_vec.size() > 1)
        std::cout << "Second common label "
                  << "\t" << label_count_vec[label_count_vec.size() - 2].first
                  << " with count=" << label_count_vec[label_count_vec.size() - 2].second << std::endl;
    if (label_count_vec.size() > 2)
        std::cout << "Third common label "
                  << "\t" << label_count_vec[label_count_vec.size() - 3].first
                  << " with count=" << label_count_vec[label_count_vec.size() - 3].second << std::endl;
    avg_labels_per_pt = sum / (float)point_cnt;
    mean_label_size = sum / (float)label_counts.size();
    std::cout << "Total number of points = " << point_cnt << ", number of labels = " << label_counts.size()
              << std::endl;
    std::cout << "Average number of labels per point = " << avg_labels_per_pt << std::endl;
    std::cout << "Mean label size excluding 0 = " << mean_label_size << std::endl;
    std::cout << "Most popular label is " << label_with_max_points << " with " << max_points << " pts" << std::endl;
}

int main(int argc, char **argv)
{
    std::string labels_file, universal_label;
    uint32_t density;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("labels_file", po::value<std::string>(&labels_file)->required(),
                           "path to labels data file.");
        desc.add_options()("universal_label", po::value<std::string>(&universal_label)->required(),
                           "Universal label used in labels file.");
        desc.add_options()("density", po::value<uint32_t>(&density)->default_value(1),
                           "Number of labels each point in labels file, defaults to 1");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }
    stats_analysis(labels_file, universal_label, density);
}
