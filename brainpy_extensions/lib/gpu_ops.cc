// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "event_sum_gpu_kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;
      dict["gpu_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_event_sum_homo_f32_i32);
      dict["gpu_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_event_sum_homo_f32_i64);
      dict["gpu_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_event_sum_homo_f64_i32);
      dict["gpu_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_event_sum_homo_f64_i64);

      dict["gpu_event_sum2_f32_i32"] = EncapsulateFunction(gpu_event_sum2_f32_i32);
      dict["gpu_event_sum2_f32_i64"] = EncapsulateFunction(gpu_event_sum2_f32_i64);
      dict["gpu_event_sum2_f64_i32"] = EncapsulateFunction(gpu_event_sum2_f64_i32);
      dict["gpu_event_sum2_f64_i64"] = EncapsulateFunction(gpu_event_sum2_f64_i64);
      return dict;
    }

    PYBIND11_MODULE(gpu_ops, m) {
      m.def("registrations", &Registrations);
      m.def("build_gpu_descriptor", [](std::int64_t size) {
        return PackDescriptor(SizeDescriptor{size});
        });
    }
}  // namespace
