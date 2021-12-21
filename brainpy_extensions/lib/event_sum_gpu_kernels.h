#ifndef _BRAINPY_event_sum_KERNELS_H_
#define _BRAINPY_event_sum_KERNELS_H_

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace brainpy_lib {
    struct SizeDescriptor {
        std::int32_t size;
    };

    void gpu_event_sum_homo_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum_homo_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    void gpu_event_sum2_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);
    void gpu_event_sum2_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}  // namespace brainpy_lib

#endif