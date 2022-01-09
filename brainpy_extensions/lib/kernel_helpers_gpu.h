// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.

#ifndef _BRAINPYLIB_KERNEL_HELPERS_CUDA_H_
#define _BRAINPYLIB_KERNEL_HELPERS_CUDA_H_

#include <cstdint>
#include <cuda_runtime_api.h>

namespace brainpy_lib {
    // error handling //
    void ThrowIfError(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
    }
  }
}  // namespace brainpy_lib

#endif
