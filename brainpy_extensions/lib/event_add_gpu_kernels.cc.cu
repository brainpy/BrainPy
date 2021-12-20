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

        template<typename F, typename I>
        __global__ void event_add_homo_kernel(const std::int32_t size,
                                         const bool *events,
                                         const I *indices,
                                         const I *indptr,
                                         const F value,
                                         F *result) {
            for (std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[i]) {
                    for (I j = indptr[i]; j < indptr[i + 1]; j++)
                        atomicAdd(&result[j], value)
                }
            }
        }

        template<typename F, typename I>
        inline void event_add_homo(cudaStream_t stream,
                              void **buffers,
                              const char *opaque,
                              std::size_t opaque_len) {
            // size
            const SizeDescriptor &d = *UnpackDescriptor<SizeDescriptor>(opaque, opaque_len);
            const std::int32_t size = d.size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F value = *reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
            event_add_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(
              size, events, indices, indptr, value, result);
            ThrowIfError(cudaGetLastError());
        }


        // "event_add2" operator //

        template<typename F, typename I>
        __global__ void event_add2_kernel(const std::int32_t size,
                                          const bool *events,
                                          const I *pre_ids,
                                          const I *post_ids,
                                          const F value,
                                          F *result) {
            for (std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[pre_ids[i]]) {
                    atomicAdd(&result[post_ids[i]], value)
                }
            }
        }

        template<typename F, typename I>
        inline void event_add2(cudaStream_t stream,
                               void **buffers,
                               const char *opaque,
                               std::size_t opaque_len) {
            // size
            const SizeDescriptor &d = *UnpackDescriptor<SizeDescriptor>(opaque, opaque_len);
            const std::int32_t size = d.size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *pre_ids = reinterpret_cast<const I *>(buffers[1]);
            const I *post_ids = reinterpret_cast<const I *>(buffers[2]);
            const F value = *reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
            event_add2_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(size,
                                                                        events,
                                                                        pre_ids,
                                                                        post_ids,
                                                                        value,
                                                                        result);
            ThrowIfError(cudaGetLastError());
        }

    }  // namespace

    void gpu_event_add_homo_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add_homo_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add_homo_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add_homo_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


    void gpu_event_add2_f32_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add2<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add2_f32_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add2<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add2_f64_i32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add2<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }
    void gpu_event_add2_f64_i64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
        event_add2<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


}  // namespace kepler_jax
