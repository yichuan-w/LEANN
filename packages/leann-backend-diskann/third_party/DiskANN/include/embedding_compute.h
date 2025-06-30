#pragma once

#include <vector>
#include <memory>

#ifdef PYBIND11_EMBEDDED
#include <pybind11/embed.h>
#else
#include <pybind11/pybind11.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace diskann
{

class PYBIND11_EXPORT EmbeddingComputer
{
  public:
    static EmbeddingComputer &getInstance()
    {
        static EmbeddingComputer instance;
        return instance;
    }

    void initialize(const std::string &model_path)
    {
        try
        {
            py::module_ sys = py::module_::import("sys");
            py::module_ os = py::module_::import("os");

            // Add the directory containing embedd_micro.py to Python path
            std::string micro_dir = "micro";
            sys.attr("path").attr("append")(micro_dir);

            // Import our module
            py::module_ embedd = py::module_::import("embedd_micro");

            // Create benchmark config
            py::object config = embedd.attr("BenchmarkConfig")(model_path, // model_path
                                                               py::list(), // empty batch_sizes
                                                               256,        // seq_length
                                                               1,          // num_runs
                                                               true,       // use_fp16
                                                               false,      // use_cuda_graphs
                                                               false       // use_flash_attention
            );

            // Create benchmark instance
            benchmark = embedd.attr("Benchmark")(config);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to initialize Python embedding computer: " + std::string(e.what()));
        }
    }

    template <typename T>
    std::vector<float> computeEmbeddings(const std::vector<T *> &points, size_t dim, size_t batch_size = 32)
    {
        try
        {
            // Convert points to numpy array
            std::vector<T> flattened_points;
            flattened_points.reserve(points.size() * dim);

            for (const auto &point : points)
            {
                flattened_points.insert(flattened_points.end(), point, point + dim);
            }

            py::array_t<T> points_array({static_cast<long>(points.size()), static_cast<long>(dim)},
                                        flattened_points.data());

            // Call compute_embeddings
            py::object result = benchmark.attr("compute_embeddings")(points_array, batch_size);

            // Convert result back to C++
            py::array_t<float> np_result = result.cast<py::array_t<float>>();
            py::buffer_info buf = np_result.request();
            float *ptr = static_cast<float *>(buf.ptr);

            return std::vector<float>(ptr, ptr + buf.size);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to compute embeddings: " + std::string(e.what()));
        }
    }

  private:
    EmbeddingComputer()
    {
#ifdef PYBIND11_EMBEDDED
        if (!Py_IsInitialized())
        {
            py::initialize_interpreter();
        }
#endif
    }

    ~EmbeddingComputer()
    {
#ifdef PYBIND11_EMBEDDED
        if (Py_IsInitialized())
        {
            py::finalize_interpreter();
        }
#endif
    }

    py::object benchmark;
};

} // namespace diskann