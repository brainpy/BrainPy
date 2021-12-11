// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "event_add_gpu_kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;
      dict["gpu_event_add_f32"] = EncapsulateFunction(gpu_event_add_f32);
      dict["gpu_event_add_f64"] = EncapsulateFunction(gpu_event_add_f64);
      dict["gpu_event_add_v2_f32"] = EncapsulateFunction(gpu_event_add_v2_f32);
      dict["gpu_event_add_v2_f64"] = EncapsulateFunction(gpu_event_add_v2_f64);
      return dict;
    }

    PYBIND11_MODULE(gpu_ops, m) {
      m.def("registrations", &Registrations);
      m.def("build_gpu_descriptor", [](std::int64_t size) {
        return PackDescriptor(SizeDescriptor{size});
        });
    }
}  // namespace
