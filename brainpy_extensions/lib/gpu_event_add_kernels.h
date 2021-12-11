#ifndef _EVENT_ADD_KERNELS_H_
#define _EVENT_ADD_KERNELS_H_

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace brainpy_lib {
    struct SizeDescriptor {
      std::int64_t size;
    };

    void gpu_event_add_f32(cudaStream_t stream,
                           void** buffers,
                           const char* opaque,
                           std::size_t opaque_len);
    void gpu_event_add_f64(cudaStream_t stream,
                           void** buffers,
                           const char* opaque,
                           std::size_t opaque_len);
    void gpu_event_add_v2_f32(cudaStream_t stream,
                              void** buffers,
                              const char* opaque,
                              std::size_t opaque_len);
    void gpu_event_add_v2_f64(cudaStream_t stream,
                              void** buffers,
                              const char* opaque,
                              std::size_t opaque_len);
}  // namespace kepler_jax

#endif