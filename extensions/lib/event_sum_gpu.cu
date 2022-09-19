// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "event_sum_gpu.h"

namespace brainpy_lib {

    namespace {


// "event_sum_homo" operator //
// This function launches "num_of_pre_neuron" threads to
// update the "result" (in global memory)
        template<typename F, typename I>
        __global__ void event_sum_homo_kernel(const std::uint32_t size,
                                              const bool *events,
                                              const I *indices,
                                              const I *indptr,
                                              const F &value,
                                              F *result) {
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
        inline void gpu_event_sum_homo(cudaStream_t stream,
                                       void **buffers,
                                       const char *opaque,
                                       std::size_t opaque_len) {
            // size
            const EventSumDescriptor &d = *UnpackDescriptor<EventSumDescriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *value = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (pre_size + block_dim - 1) / block_dim);
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(pre_size, events, indices, indptr, value[0],
                                                                            result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void event_sum_heter_kernel(const std::uint32_t size,
                                               const bool *events,
                                               const I *indices,
                                               const I *indptr,
                                               const F *values,
                                               F *result) {
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
        inline void gpu_event_sum_heter(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
            // size
            const EventSumDescriptor &d = *UnpackDescriptor<EventSumDescriptor>(opaque, opaque_len);
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
            const int grid_dim = std::min<int>(1024, (pre_size + block_dim - 1) / block_dim);
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(pre_size, events, indices, indptr, values,
                                                                             result);
            ThrowIfError(cudaGetLastError());
        }


// "event_sum2" operator //

        template<typename F, typename I>
        __global__ void event_sum2_homo_kernel(const std::uint32_t size,
                                               const bool *events,
                                               const I *pre_ids,
                                               const I *post_ids,
                                               const F &value,
                                               F *result) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[pre_ids[i]]) {
                    atomicAdd(&result[post_ids[i]], value);
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_event_sum2_homo(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
            // size
            const EventSum2Descriptor &d = *UnpackDescriptor<EventSum2Descriptor>(opaque, opaque_len);
            const std::uint32_t conn_size = d.conn_size;
            const std::uint32_t post_size = d.post_size;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *pre_ids = reinterpret_cast<const I *>(buffers[1]);
            const I *post_ids = reinterpret_cast<const I *>(buffers[2]);
            const F *value = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 512;
            const int grid_dim = std::min<int>(1024, (conn_size + block_dim - 1) / block_dim);
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum2_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, events, pre_ids, post_ids,
                                                                             value[0], result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void event_sum2_heter_kernel(const std::uint32_t size,
                                                const bool *events,
                                                const I *pre_ids,
                                                const I *post_ids,
                                                const F *values,
                                                F *result) {
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < size; i += blockDim.x * gridDim.x) {
                if (events[pre_ids[i]]) {
                    atomicAdd(&result[post_ids[i]], values[i]);
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_event_sum2_heter(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
            // size
            const EventSum2Descriptor &d = *UnpackDescriptor<EventSum2Descriptor>(opaque, opaque_len);
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
            const int grid_dim = std::min<int>(1024, (conn_size + block_dim - 1) / block_dim);
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum2_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, events, pre_ids, post_ids,
                                                                              values, result);
            ThrowIfError(cudaGetLastError());
        }




        // The third method to make "event_sum" //
        // This method is inspired by GeNN codes.

        __global__ void collect_spike_info(const bool *events,
                                           const std::uint32_t pre_size,
                                           unsigned int *event_ids,
                                           unsigned int *event_num) {
            const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
//             __shared__ unsigned int shSpk[blockDim.x];
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

        template<typename F, typename I>
        __global__ void event_sum3_homo_kernel(const std::uint32_t max_post_num,
                                               const I *indices,
                                               const I *indptr,
                                               const F *values,
                                               const unsigned int *event_ids,
                                               const unsigned int *event_num,
                                               F *result) {
            const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
//            __shared__ unsigned int shSpk[blockDim.x];
//             __shared__ I shPreStartID[blockDim.x];
//             __shared__ I shRowLength[blockDim.x];
            __shared__ I shPreStartID[32];
            __shared__ I shRowLength[32];
            __shared__ unsigned int event_count;
            __shared__ F value;

            if (threadIdx.x == 0) {
                value = values[0];
                event_count = event_num[0];
            }
            __syncthreads();

            if (id < max_post_num) {
                const unsigned int num_iter = (event_count + blockDim.x - 1) / blockDim.x;
                for (unsigned int r = 0; r < num_iter; r++) {
                    const unsigned int num_event = (r == num_iter - 1) ? ((event_count - 1) % blockDim.x) + 1
                                                                       : blockDim.x;
                    __syncthreads();
                    if (threadIdx.x < num_event) {
                        const unsigned int pre_i = event_ids[(r * 32) + threadIdx.x];
//                        shSpk[threadIdx.x] = pre_i;
//                        shRowLength[threadIdx.x] = indptr[pre_i + 1] - indptr[pre_i];
                        shPreStartID[threadIdx.x] = indptr[pre_i];
                        shRowLength[threadIdx.x] = indptr[pre_i + 1] - shPreStartID[threadIdx.x];
                    }
                    __syncthreads();
                    // loop through all incoming spikes
                    for (unsigned int j = 0; j < num_event; j++) {
                        // only work on existing neurons
                        const I post_num = shRowLength[j];
                        if (id < post_num) {
//                            const I post_i = indices[indptr[shSpk[j]] + id];
                            const I post_i = indices[shPreStartID[j] + id];
                            atomicAdd(&result[post_i], value);
                        }
                    }
                }
            }
        }


        template<typename F, typename I>
        inline void gpu_event_sum3_homo(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
            // size information
            const EventSum3Descriptor &d = *UnpackDescriptor<EventSum3Descriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;
            const std::uint32_t max_post_conn = d.max_post_conn;

            // input and output data //
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // get spike information //
            unsigned int *event_ids;
            cudaMalloc(&event_ids, pre_size * sizeof(unsigned int));
            // I *spikes[pre_size];
            // cudaMemset(spikes, 0, sizeof(I)*pre_size);
            unsigned int *event_num;
            cudaMalloc(&event_num, 1 * sizeof(unsigned int));
            int block_dim = 64;
            int grid_dim = (pre_size + block_dim - 1) / block_dim;
            collect_spike_info<<<grid_dim, block_dim, 0, stream>>>(events,
                                                                   pre_size,
                                                                   event_ids,
                                                                   event_num);

            // event sum kernel //
            cudaMemset(result, 0, sizeof(F) * post_size);
            block_dim = 32;
            grid_dim = (max_post_conn + block_dim - 1) / block_dim;
            event_sum3_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(max_post_conn,
                                                                             indices,
                                                                             indptr,
                                                                             values,
                                                                             event_ids,
                                                                             event_num,
                                                                             result);

            // free memory
            cudaFree(event_ids);
            cudaFree(event_num);

            // check error
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void event_sum3_heter_kernel(const std::uint32_t max_post_num,
                                                const I *indices,
                                                const I *indptr,
                                                const F *values,
                                                const unsigned int *event_ids,
                                                const unsigned int *event_num,
                                                F *result) {
            const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
//            __shared__ unsigned int shSpk[blockDim.x];
//             __shared__ I shPreStartID[blockDim.x];
//             __shared__ I shRowLength[blockDim.x];
            __shared__ I shPreStartID[32];
            __shared__ I shRowLength[32];
            __shared__ unsigned int event_count;

            if (threadIdx.x == 0) {
                event_count = event_num[0];
            }
            __syncthreads();

            if (id < max_post_num) {
                const unsigned int num_iter = (event_count + blockDim.x - 1) / blockDim.x;
                for (unsigned int r = 0; r < num_iter; r++) {
                    const unsigned int num_event = (r == num_iter - 1) ? ((event_count - 1) % blockDim.x) + 1
                                                                       : blockDim.x;
                    __syncthreads();
                    if (threadIdx.x < num_event) {
                        const unsigned int spk = event_ids[(r * 32) + threadIdx.x];
//                        shSpk[threadIdx.x] = spk;
//                        shRowLength[threadIdx.x] = indptr[spk + 1] - indptr[spk];
                        shPreStartID[threadIdx.x] = indptr[spk];
                        shRowLength[threadIdx.x] = indptr[spk + 1] - shPreStartID[threadIdx.x];
                    }
                    __syncthreads();
                    // loop through all incoming spikes
                    for (unsigned int j = 0; j < num_event; j++) {
                        // only work on existing neurons
                        const I post_num = shRowLength[j];
                        if (id < post_num) {
//                            const I syn_i = indptr[shSpk[j]] + id;
                            const I syn_i = shPreStartID[j] + id;
                            const I post_i = indices[syn_i];
                            atomicAdd(&result[post_i], values[syn_i]);
                        }
                    }
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_event_sum3_heter(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
            // size information
            const EventSum3Descriptor &d = *UnpackDescriptor<EventSum3Descriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;
            const std::uint32_t max_post_conn = d.max_post_conn;

            // input and output data //
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]);
            F *result = reinterpret_cast<F *>(buffers[4]);

            // get spike information //
            unsigned int *event_ids;
            cudaMalloc(&event_ids, pre_size * sizeof(unsigned int));
            // I *spikes[pre_size];
            // cudaMemset(spikes, 0, sizeof(I)*pre_size);
            unsigned int *event_num;
            cudaMalloc(&event_num, 1 * sizeof(unsigned int));
            int block_dim = 64;
            int grid_dim = (pre_size + block_dim - 1) / block_dim;
            collect_spike_info<<<grid_dim, block_dim, 0, stream>>>(events,
                                                                   pre_size,
                                                                   event_ids,
                                                                   event_num);

            // event sum kernel //
            cudaMemset(result, 0, sizeof(F) * post_size);
            block_dim = 32;
            grid_dim = (max_post_conn + block_dim - 1) / block_dim;
            event_sum3_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(max_post_conn,
                                                                              indices,
                                                                              indptr,
                                                                              values,
                                                                              event_ids,
                                                                              event_num,
                                                                              result);

            // free memory
            cudaFree(event_ids);
            cudaFree(event_num);

            // check error
            ThrowIfError(cudaGetLastError());
        }


        template<typename F, typename I>
        __global__ void event_sum4_homo_kernel(const std::uint32_t max_post_conn,
                                               const std::uint32_t pre_size,
                                               const bool *events,
                                               const I *indices,
                                               const I *indptr,
                                               const F *values,
                                               F *result) {
            __shared__ bool shared_events[32];
            __shared__ I shRowLength[32];
            __shared__ I shPreStartID[32];
            __shared__ F value;

            if (threadIdx.x == 0) {
                value = values[0];
            }
            __syncthreads();

            const I id = blockIdx.x * 32 + threadIdx.x;
            if (id < max_post_conn) {
                const unsigned int num_iter = (pre_size + 32 - 1) / 32;
                for (unsigned int r = 0; r < num_iter; r++) {
                    const unsigned int num_event = (r == num_iter - 1) ? ((pre_size - 1) % 32) + 1 : 32;
                    // assume "max_post_conn" >= num_event
                    if (threadIdx.x < num_event) {
                        const unsigned int pre_i = (r * 32) + threadIdx.x;
                        shared_events[threadIdx.x] = events[pre_i];
                        if (shared_events[threadIdx.x])
                        {
                            shPreStartID[threadIdx.x] = indptr[pre_i];
                            shRowLength[threadIdx.x] = indptr[pre_i + 1] - shPreStartID[threadIdx.x];
                        }
                    }
                    __syncthreads();
                    for (unsigned int j = 0; j < num_event; j++) {
                        if (shared_events[j]) {
                            if (id < shRowLength[j]) {
                                const I syn_i = shPreStartID[j] + id;
                                const I post_i = indices[syn_i];
                                atomicAdd(&result[post_i], value);
                            }
                        }
                    }
                }
            }
        }

        template<typename F, typename I>
        inline void gpu_event_sum4_homo(cudaStream_t stream,
                                        void **buffers,
                                        const char *opaque,
                                        std::size_t opaque_len) {
            // size
            const EventSum3Descriptor &d = *UnpackDescriptor<EventSum3Descriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;
            const std::uint32_t max_post_conn = d.max_post_conn;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]); // 1D vector with the size of 1
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 32;
            const int grid_dim = (max_post_conn + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum4_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(max_post_conn,
                                                                             pre_size,
                                                                             events,
                                                                             indices,
                                                                             indptr,
                                                                             values,
                                                                             result);
            ThrowIfError(cudaGetLastError());
        }

        template<typename F, typename I>
        __global__ void event_sum4_heter_kernel(const std::uint32_t max_post_conn,
                                                const std::uint32_t pre_size,
                                                const bool *events,
                                                const I *indices,
                                                const I *indptr,
                                                const F *values,
                                                F *result) {
            __shared__ bool shared_events[32];
            __shared__ I shRowLength[32];
            __shared__ I shPreStartID[32];

            const I id = blockIdx.x * 32 + threadIdx.x;
            if (id < max_post_conn) {
                const unsigned int num_iter = (pre_size + 32 - 1) / 32;
                for (unsigned int r = 0; r < num_iter; r++) {
                    const unsigned int num_event = (r == num_iter - 1) ? ((pre_size - 1) % 32) + 1 : 32;
                    // assume "max_post_conn" >= num_event
                    // TODO: fix the bug
                    if (threadIdx.x < num_event) {
                        const unsigned int pre_i = (r * 32) + threadIdx.x;
                        shared_events[threadIdx.x] = events[pre_i];
                        if (shared_events[threadIdx.x])
                        {
                            shPreStartID[threadIdx.x] = indptr[pre_i];
                            shRowLength[threadIdx.x] = indptr[pre_i + 1] - shPreStartID[threadIdx.x];
                        }
                    }
                    __syncthreads();
                    for (unsigned int j = 0; j < num_event; j++) {
                        if (shared_events[j]) {
                            if (id < shRowLength[j]) {
                                const I syn_i = shPreStartID[j] + id;
                                const I post_i = indices[syn_i];
                                atomicAdd(&result[post_i], values[syn_i]);
                            }
                        }
                    }
                }
            }
        }



        template<typename F, typename I>
        inline void gpu_event_sum4_heter(cudaStream_t stream,
                                         void **buffers,
                                         const char *opaque,
                                         std::size_t opaque_len) {
            // size
            const EventSum3Descriptor &d = *UnpackDescriptor<EventSum3Descriptor>(opaque, opaque_len);
            const std::uint32_t pre_size = d.pre_size;
            const std::uint32_t post_size = d.post_size;
            const std::uint32_t max_post_conn = d.max_post_conn;

            // input and output data
            const bool *events = reinterpret_cast<const bool *>(buffers[0]);
            const I *indices = reinterpret_cast<const I *>(buffers[1]);
            const I *indptr = reinterpret_cast<const I *>(buffers[2]);
            const F *values = reinterpret_cast<const F *>(buffers[3]); // 1D vector with the size of 1
            F *result = reinterpret_cast<F *>(buffers[4]);

            // call kernel
            const int block_dim = 32;
            const int grid_dim = (max_post_conn + block_dim - 1) / block_dim;
            cudaMemset(result, 0, sizeof(F) * post_size);
            event_sum4_heter_kernel<F, I><<<grid_dim, block_dim,
            /*dynamic_shared_mem_bytes=*/0, stream>>>(max_post_conn,
                                                                              pre_size,
                                                                              events,
                                                                              indices,
                                                                              indptr,
                                                                              values,
                                                                              result);
            ThrowIfError(cudaGetLastError());
        }



    }  // namespace


    // Descriptors
    pybind11::bytes build_event_sum_descriptor(std::uint32_t pre_size,
                                               std::uint32_t post_size) {
        return PackDescriptor(EventSumDescriptor{pre_size, post_size});
    }

    pybind11::bytes build_event_sum2_descriptor(std::uint32_t conn_size,
                                                std::uint32_t post_size) {
        return PackDescriptor(EventSum2Descriptor{conn_size, post_size});
    }

    pybind11::bytes build_event_sum3_descriptor(std::uint32_t pre_size,
                                                std::uint32_t post_size,
                                                std::uint32_t max_post_conn) {
        return PackDescriptor(EventSum3Descriptor{pre_size, post_size, max_post_conn});
    }


    // homogenous event sum 1
    void gpu_event_sum_homo_f32_i32(cudaStream_t stream,
                                    void **buffers,
                                    const char *opaque,
                                    std::size_t opaque_len) {
        gpu_event_sum_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_homo_f32_i64(cudaStream_t stream,
                                    void **buffers,
                                    const char *opaque,
                                    std::size_t opaque_len) {
        gpu_event_sum_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_homo_f64_i32(cudaStream_t stream,
                                    void **buffers,
                                    const char *opaque,
                                    std::size_t opaque_len) {
        gpu_event_sum_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_homo_f64_i64(cudaStream_t stream,
                                    void **buffers,
                                    const char *opaque,
                                    std::size_t opaque_len) {
        gpu_event_sum_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    // heterogeneous event sum 1
    void gpu_event_sum_heter_f32_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_heter_f32_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_heter_f64_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum_heter_f64_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


    // homogenous event sum 2
    void gpu_event_sum2_homo_f32_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum2_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_homo_f32_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum2_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_homo_f64_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum2_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_homo_f64_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum2_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    // heterogeneous event sum 2
    void gpu_event_sum2_heter_f32_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum2_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_heter_f32_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum2_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_heter_f64_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum2_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum2_heter_f64_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum2_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


    // homogenous event sum 3
    void gpu_event_sum3_homo_f32_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum3_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_homo_f32_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum3_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_homo_f64_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum3_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_homo_f64_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum3_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    // heterogeneous event sum 3
    void gpu_event_sum3_heter_f32_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum3_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_heter_f32_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum3_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_heter_f64_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum3_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum3_heter_f64_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum3_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


    // homogenous event sum 3
    void gpu_event_sum4_homo_f32_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum4_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_homo_f32_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum4_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_homo_f64_i32(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum4_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_homo_f64_i64(cudaStream_t stream,
                                     void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
        gpu_event_sum4_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    // heterogeneous event sum 3
    void gpu_event_sum4_heter_f32_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum4_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_heter_f32_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum4_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_heter_f64_i32(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum4_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
    }

    void gpu_event_sum4_heter_f64_i64(cudaStream_t stream,
                                      void **buffers,
                                      const char *opaque,
                                      std::size_t opaque_len) {
        gpu_event_sum4_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
    }


}  // namespace brainpylib
