// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point across.

#include "event_sum_gpu.h"

namespace brainpy_lib {

namespace {


// "event_sum" operator //
template<typename F, typename I>
__global__ void gpu_event_sum_homo_kernel(const std::uint32_t size,
                                          const bool *events,
                                          const I *indices,
                                          const I *indptr,
                                          const F &value,
                                          F *result) {
    for (std::uint32_t i=blockIdx.x * blockDim.x + threadIdx.x;
         i<size; i+=blockDim.x * gridDim.x) {
        if (events[i]) {
            for (I j=indptr[i]; j<indptr[i + 1]; ++j){
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
    cudaMemset(result, 0, sizeof(F)*post_size);
    gpu_event_sum_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(pre_size, events, indices, indptr, value[0], result);
    ThrowIfError(cudaGetLastError());
}

template<typename F, typename I>
__global__ void gpu_event_sum_heter_kernel(const std::uint32_t size,
                                           const bool *events,
                                           const I *indices,
                                           const I *indptr,
                                           const F *values,
                                           F *result) {
    for (std::uint32_t i=blockIdx.x * blockDim.x + threadIdx.x;
         i<size; i+=blockDim.x * gridDim.x) {
        if (events[i]) {
            for (I j=indptr[i]; j<indptr[i + 1]; ++j){
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
    cudaMemset(result, 0, sizeof(F)*post_size);
    gpu_event_sum_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(pre_size, events, indices, indptr, values, result);
    ThrowIfError(cudaGetLastError());
}


// "event_sum2" operator //

template<typename F, typename I>
__global__ void gpu_event_sum2_homo_kernel(const std::uint32_t size,
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
    cudaMemset(result, 0, sizeof(F)*post_size);
    gpu_event_sum2_homo_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, events, pre_ids, post_ids, value[0], result);
    ThrowIfError(cudaGetLastError());
}

template<typename F, typename I>
__global__ void gpu_event_sum2_heter_kernel(const std::uint32_t size,
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

    // input and output data
    const bool *events = reinterpret_cast<const bool *>(buffers[0]);
    const I *pre_ids = reinterpret_cast<const I *>(buffers[1]);
    const I *post_ids = reinterpret_cast<const I *>(buffers[2]);
    const F *values = reinterpret_cast<const F *>(buffers[3]);
    F *result = reinterpret_cast<F *>(buffers[4]);

    // call kernel
    const int block_dim = 512;
    const int grid_dim = std::min<int>(1024, (conn_size + block_dim - 1) / block_dim);
    cudaMemset(result, 0, sizeof(F)*post_size);
    gpu_event_sum2_heter_kernel<F, I><<<grid_dim, block_dim, 0, stream>>>(conn_size, events, pre_ids, post_ids, values, result);
    ThrowIfError(cudaGetLastError());
}




}  // namespace


// Descriptor

pybind11::bytes build_event_sum_descriptor(std::uint32_t pre_size,
                                              std::uint32_t post_size){
    return PackDescriptor(EventSumDescriptor{pre_size, post_size});
}

pybind11::bytes build_event_sum2_descriptor(std::uint32_t conn_size,
                                                std::uint32_t post_size){
    return PackDescriptor(EventSum2Descriptor{conn_size, post_size});
}


// homogenous event sum 1
void gpu_event_sum_homo_f32_i32(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len){
    gpu_event_sum_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_homo_f32_i64(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len){
    gpu_event_sum_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_homo_f64_i32(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len){
    gpu_event_sum_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_homo_f64_i64(cudaStream_t stream, void **buffers,
                                const char *opaque, std::size_t opaque_len){
    gpu_event_sum_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
}

// heterogeneous event sum 1
void gpu_event_sum_heter_f32_i32(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_heter_f32_i64(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_heter_f64_i32(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum_heter_f64_i64(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
}


// homogenous event sum 2
void gpu_event_sum2_homo_f32_i32(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_homo<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_homo_f32_i64(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_homo<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_homo_f64_i32(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_homo<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_homo_f64_i64(cudaStream_t stream, void **buffers,
                                 const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_homo<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
}
// heterogeneous event sum 2
void gpu_event_sum2_heter_f32_i32(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_heter<float, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_heter_f32_i64(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_heter<float, std::uint64_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_heter_f64_i32(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_heter<double, std::uint32_t>(stream, buffers, opaque, opaque_len);
}
void gpu_event_sum2_heter_f64_i64(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len){
    gpu_event_sum2_heter<double, std::uint64_t>(stream, buffers, opaque, opaque_len);
}


}  // namespace brainpylib
