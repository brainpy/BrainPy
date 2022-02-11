// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "atomic_prod_gpu.h"

namespace brainpy_lib {

    namespace {

// "atomic_prod" operator //
        template<typename F, typename I>
        __global__ void gpu_atomic_prod_homo_kernel(const std::uint32_t size,
                                                   const F &value,
                                                   const I *post_ids,
                                                   F *result) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                atomicAdd(&result[post_ids[i]], value);
            }
        }

        template<typename F, typename I>
        inline void gpu_atomic_prod_homo(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
            // size
            const AtomicProdDescriptor &d = *UnpackDescriptor<AtomicProdDescriptor>(opaque, opaque_len);
            const std::uint32_t conn_size = d.conn_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const F *values = reinterpret_cast<const F *>(buffers[0]);  // scalar as a vector
            const I *post_ids = reinterpret_cast<const I *>(buffers[1]);
            F *result = reinterpret_cast<F *>(buffers[2]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (conn_size + block_dim - 1) / block_dim);
            cudaMemset(result, 1, sizeof(F) * post_size);
            gpu_atomic_prod_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, values[0], post_ids,
                                                                                 result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void gpu_atomic_prod_heter_kernel(const std::uint32_t size,
                                                    const F *values,
                                                    const I *post_ids,
                                                    const I *pre_ids,
                                                    F *result) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                atomicAdd(&result[post_ids[i]], values[pre_ids[i]]);
            }
        }

        template<typename F, typename I>
        inline void gpu_atomic_prod_heter(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
            // size
            const AtomicProdDescriptor &d = *UnpackDescriptor<AtomicProdDescriptor>(opaque, opaque_len);
            const std::uint32_t conn_size = d.conn_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const F *values = reinterpret_cast<const F *>(buffers[0]);  // scalar as a vector
            const I *post_ids = reinterpret_cast<const I *>(buffers[1]);
            const I *pre_ids = reinterpret_cast<const I *>(buffers[2]);
            F *result = reinterpret_cast<F *>(buffers[3]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (conn_size + block_dim - 1) / block_dim);
            cudaMemset(result, 1, sizeof(F) * post_size);
            gpu_atomic_prod_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, values, post_ids, pre_ids,
                                                                                  result);
            ThrowIfError(cudaGetLastError());
        }


    }  // namespace


// Descriptor
    pybind11::bytes build_atomic_prod_descriptor(std::uint32_t conn_size,
                                                std::uint32_t post_size) {
        return PackDescriptor(AtomicProdDescriptor{conn_size, post_size});
    }

// homogenous atomic sum
    void gpu_atomic_prod_homo_f32_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_atomic_prod_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_homo_f32_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_atomic_prod_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_homo_f64_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_atomic_prod_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_homo_f64_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_atomic_prod_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

// heterogeneous atomic sum
    void gpu_atomic_prod_heter_f32_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_atomic_prod_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_heter_f32_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_atomic_prod_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_heter_f64_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_atomic_prod_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_atomic_prod_heter_f64_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_atomic_prod_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


}  // namespace brainpylib
