// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "gpu_event_sum.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


namespace brainpy_lib {

    namespace {

        // "event_sum_homo" operator //
        // This function launches "num_of_pre_neuron" threads to
        // update the "result" (in global memory)
        template<typename F, typename I>
        __global__ void _csr_event_sum_homo_kernel(
                const std::uint32_t size,
                const bool *events,
                const I *indices,
                const I *indptr,
                const F &value,
                F *result
        ) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[i]) {
                    for (I j = indptr[i]; j < indptr[i + 1]; ++j) {
                        atomicAdd(&result[indices[j]], value);
                    }
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_csr_event_sum_homo(cudaStream_t stream,
                                           void **buffers,
                                           const char *opaque,
                                           std::size_t opaque_len) {
            // size
            const CSREventSumDescriptor &d = *UnpackDescriptor<CSREventSumDescriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *weights = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = (pre_size + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            _csr_event_sum_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(
                    pre_size, events, indices, indptr, weights[0], result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void _csr_event_sum_heter_kernel(
                const std::uint32_t size,
                const bool *events,
                const I *indices,
                const I *indptr,
                const F *values,
                F *result
        ) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[i]) {
                    for (I j = indptr[i]; j < indptr[i + 1]; ++j) {
                        atomicAdd(&result[indices[j]], values[j]);
                    }
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_csr_event_sum_heter(cudaStream_t stream,
                                            void **buffers,
                                            const char *opaque,
                                            std::size_t opaque_len) {
            // size
            const CSREventSumDescriptor &d = *UnpackDescriptor<CSREventSumDescriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = (pre_size + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            _csr_event_sum_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(
                    pre_size, events, indices, indptr, values, result);
            ThrowIfError(cudaGetLastError());
        }


// "event_sum2" operator //
        template<typename F, typename I>
        __global__ void _coo_event_sum_homo_kernel(
                const std::uint32_t size,
                const bool *events,
                const I *pre_ids,
                const I *post_ids,
                const F &value,
                F *result
        ) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[pre_ids[i]]) {
                    atomicAdd(&result[post_ids[i]], value);
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_coo_event_sum_homo(cudaStream_t stream,
                                           void **buffers,
                                           const char *opaque,
                                           std::size_t opaque_len) {
            // size
            const COOEventSumDescriptor &d = *UnpackDescriptor<COOEventSumDescriptor>(opaque, opaque_len);
            const std::uint32_t conn_size = d.conn_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *pre_ids = reinterpret_cast<const I *>(buffers[1]);
            const I *post_ids = reinterpret_cast<const I *>(buffers[2]);
            const F *weights = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = (conn_size + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            _coo_event_sum_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(
                    conn_size, events, pre_ids, post_ids, weights[0], result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void _coo_event_sum_heter_kernel(
                const std::uint32_t size,
                const bool *events,
                const I *pre_ids,
                const I *post_ids,
                const F *values,
                F *result
        ) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[pre_ids[i]]) {
                    atomicAdd(&result[post_ids[i]], values[i]);
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_coo_event_sum_heter(cudaStream_t stream,
                                            void **buffers,
                                            const char *opaque,
                                            std::size_t opaque_len) {
            // size
            const COOEventSumDescriptor &d = *UnpackDescriptor<COOEventSumDescriptor>(opaque, opaque_len);
            const std::uint32_t conn_size = d.conn_size;
            const std::uint32_t post_size = d.post_size;

            // iput and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *pre_ids = reinterpret_cast<const I *>(buffers[1]);
            const I *post_ids = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = (conn_size + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            _coo_event_sum_heter_kernel < F, I ><<<grid_dim, block_dim, 0, stream>>>(
                    conn_size, events, pre_ids, post_ids, values, result);
            ThrowIfError(cudaGetLastError());
        }




        // The third method to make "event_sum" //
        // This method is inspired by GeNN codes.

        __global__ void collect_spike_info(const bool *events,
                                           const std::uint32_t pre_size,
                                           unsigned int *event_ids,
                                           unsigned int *event_num) {
            const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
            __shared__ unsigned int shSpk[64];
            __shared__ unsigned int shPosSpk;
            __shared__ unsigned int shSpkCount;
            if (threadIdx.x == 0) {
                shSpkCount = 0;
            }
            __syncthreads();

            if (id < pre_size) {
                if (events[id]) {
                    const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                    shSpk[spkIdx] = id;
                }
                __syncthreads();

                if (threadIdx.x == 0) {
                    if (shSpkCount > 0) {
                        shPosSpk = atomicAdd(&event_num[0], shSpkCount);
                    }
                }
                __syncthreads();

                if (threadIdx.x < shSpkCount) {
                    const unsigned int n = shSpk[threadIdx.x];
                    event_ids[shPosSpk + threadIdx.x] = n;
                }
            }
        }


    }  // namespace


    // homogenous event sum 1
    pybind11::bytes build_csr_event_sum_descriptor(std::uint32_t pre_size,
                                                   std::uint32_t post_size) {
        return PackDescriptor(CSREventSumDescriptor{pre_size, post_size});
    }


    void gpu_csr_event_sum_homo_f32_i32(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_csr_event_sum_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_homo_f32_i64(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_csr_event_sum_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_homo_f64_i32(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_csr_event_sum_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_homo_f64_i64(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_csr_event_sum_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


    // heterogeneous event sum 1
    void gpu_csr_event_sum_heter_f32_i32(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_csr_event_sum_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_heter_f32_i64(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_csr_event_sum_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_heter_f64_i32(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_csr_event_sum_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_csr_event_sum_heter_f64_i64(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_csr_event_sum_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }



    // homogenous event sum 2
    pybind11::bytes build_coo_event_sum_descriptor(std::uint32_t conn_size,
                                                   std::uint32_t post_size) {
        return PackDescriptor(COOEventSumDescriptor{conn_size, post_size});
    }

    void gpu_coo_event_sum_homo_f32_i32(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_coo_event_sum_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_homo_f32_i64(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_coo_event_sum_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_homo_f64_i32(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_coo_event_sum_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_homo_f64_i64(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
        gpu_coo_event_sum_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    // heterogeneous event sum 2
    void gpu_coo_event_sum_heter_f32_i32(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_coo_event_sum_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_heter_f32_i64(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_coo_event_sum_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_heter_f64_i32(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_coo_event_sum_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_coo_event_sum_heter_f64_i64(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
        gpu_coo_event_sum_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


}  // namespace brainpylib
