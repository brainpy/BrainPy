// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kernel_helpers.h"
#include "event_add_gpu_kernels.h"

namespace brainpy_lib {

  namespace {

    // error handling //
    void ThrowIfError(cudaError_t error) {
      if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }


    // "event_add" operator //

    template <typename T>
    __global__ void event_add_kernel(std::int64_t size,
                                     const bool *events,
                                     const std::int64_t *indices,
                                     const std::int64_t *indptr,
                                     const T value,
                                     T *result) {
      for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < size; i += blockDim.x * gridDim.x) {
        if (events[i]){
          for(std::int64_t j = indptr[i]; j < indptr[i+1]; j++)
            atomicAdd(&result[j], value)
        }
      }
    }

    template <typename T>
    inline void event_add(cudaStream_t stream,
                          void **buffers,
                          const char *opaque,
                          std::size_t opaque_len) {
      // size
      const SizeDescriptor &d = *UnpackDescriptor<SizeDescriptor>(opaque, opaque_len);
      const std::int64_t size = d.size;

      // input and output data
      const bool *events = reinterpret_cast<const bool *>(buffers[0]);
      const std::int64_t *indices = reinterpret_cast<const std::int64_t *>(buffers[1]);
      const std::int64_t *indptr = reinterpret_cast<const std::int64_t *>(buffers[2]);
      const T value = *reinterpret_cast<const T *>(buffers[3]);
      T *result = reinterpret_cast<T *>(buffers[4]);

      // call kernel
      const int block_dim = 512;
      const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
      event_add_kernel<T><<<grid_dim, block_dim, 0, stream>>>(size,
                                                              events,
                                                              indices,
                                                              indptr,
                                                              value,
                                                              result);
      ThrowIfError(cudaGetLastError());
    }


    // "event_add_v2" operator //

    template <typename T>
    __global__ void event_add_v2_kernel(std::int64_t size,
                                        const bool *events,
                                        const std::int64_t *pre_ids,
                                        const std::int64_t *post_ids,
                                        const T value,
                                        T *result) {
      for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < size; i += blockDim.x * gridDim.x) {
        if (events[pre_ids[i]]){
            atomicAdd(&result[post_ids[i]], value)
        }
      }
    }

    template <typename T>
    inline void event_add_v2(cudaStream_t stream,
                             void **buffers,
                             const char *opaque,
                             std::size_t opaque_len) {
      // size
      const SizeDescriptor &d = *UnpackDescriptor<SizeDescriptor>(opaque, opaque_len);
      const std::int64_t size = d.size;

      // input and output data
      const bool *events = reinterpret_cast<const bool *>(buffers[0]);
      const std::int64_t *pre_ids = reinterpret_cast<const std::int64_t *>(buffers[1]);
      const std::int64_t *post_ids = reinterpret_cast<const std::int64_t *>(buffers[2]);
      const T value = *reinterpret_cast<const T *>(buffers[3]);
      T *result = reinterpret_cast<T *>(buffers[4]);

      // call kernel
      const int block_dim = 512;
      const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
      event_add_v2_kernel<T><<<grid_dim, block_dim, 0, stream>>>(size,
                                                                 events,
                                                                 pre_ids,
                                                                 post_ids,
                                                                 value,
                                                                 result);
      ThrowIfError(cudaGetLastError());
    }

  }  // namespace

void gpu_event_add_f32(cudaStream_t stream,
                       void **buffers,
                       const char *opaque,
                       std::size_t opaque_len) {
  event_add<float>(stream, buffers, opaque, opaque_len);
}

void gpu_event_add_f64(cudaStream_t stream,
                       void **buffers,
                       const char *opaque,
                       std::size_t opaque_len) {
  event_add<double>(stream, buffers, opaque, opaque_len);
}

void gpu_event_add_v2_f32(cudaStream_t stream,
                          void **buffers,
                          const char *opaque,
                          std::size_t opaque_len) {
  event_add_v2<float>(stream, buffers, opaque, opaque_len);
}

void gpu_event_add_v2_f64(cudaStream_t stream,
                          void **buffers,
                          const char *opaque,
                          std::size_t opaque_len) {
  event_add_v2<double>(stream, buffers, opaque, opaque_len);
}

}  // namespace kepler_jax
