#ifndef _BRAINPY_EVENT_SUM_KERNELS_H_
#define _BRAINPY_EVENT_SUM_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include "pybind11_kernel_helpers.h"
#include "kernel_helpers_gpu.h"

namespace brainpy_lib {
    struct EventSumDescriptor {
        std::uint32_t pre_size;
        std::uint32_t post_size;
    };

    struct EventSum2Descriptor {
        std::uint32_t conn_size;
        std::uint32_t post_size;
    };

    // homogeneous
    void gpu_event_sum_homo_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    // heterogeneous
    void gpu_event_sum_heter_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_heter_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_heter_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_heter_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    // homogeneous
    void gpu_event_sum2_homo_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_homo_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_homo_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_homo_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    // heterogeneous
    void gpu_event_sum2_heter_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_heter_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_heter_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_heter_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    // descriptors
    pybind11::bytes build_event_sum_descriptor(std::uint32_t pre_size, std::uint32_t post_size);
    pybind11::bytes build_event_sum2_descriptor(std::uint32_t conn_size, std::uint32_t post_size);

}  // namespace brainpy_lib

#endif