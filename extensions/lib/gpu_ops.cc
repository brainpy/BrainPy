// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include "pybind11_kernel_helpers.h"
#include "event_sum_gpu.h"
#include "atomic_sum_gpu.h"
#include "atomic_prod_gpu.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
        pybind11::dict dict;

        // homogeneous event_sum
        dict["gpu_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_event_sum_homo_f32_i32);
        dict["gpu_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_event_sum_homo_f32_i64);
        dict["gpu_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_event_sum_homo_f64_i32);
        dict["gpu_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_event_sum_homo_f64_i64);
        // heterogeneous event_sum
        dict["gpu_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_event_sum_heter_f32_i32);
        dict["gpu_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_event_sum_heter_f32_i64);
        dict["gpu_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_event_sum_heter_f64_i32);
        dict["gpu_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_event_sum_heter_f64_i64);

        // homogeneous event_sum2
        dict["gpu_event_sum2_homo_f32_i32"] = EncapsulateFunction(gpu_event_sum2_homo_f32_i32);
        dict["gpu_event_sum2_homo_f32_i64"] = EncapsulateFunction(gpu_event_sum2_homo_f32_i64);
        dict["gpu_event_sum2_homo_f64_i32"] = EncapsulateFunction(gpu_event_sum2_homo_f64_i32);
        dict["gpu_event_sum2_homo_f64_i64"] = EncapsulateFunction(gpu_event_sum2_homo_f64_i64);
        // heterogeneous event_sum2
        dict["gpu_event_sum2_heter_f32_i32"] = EncapsulateFunction(gpu_event_sum2_heter_f32_i32);
        dict["gpu_event_sum2_heter_f32_i64"] = EncapsulateFunction(gpu_event_sum2_heter_f32_i64);
        dict["gpu_event_sum2_heter_f64_i32"] = EncapsulateFunction(gpu_event_sum2_heter_f64_i32);
        dict["gpu_event_sum2_heter_f64_i64"] = EncapsulateFunction(gpu_event_sum2_heter_f64_i64);

        // homogeneous atomic_sum
        dict["gpu_atomic_sum_homo_f32_i32"] = EncapsulateFunction(gpu_atomic_sum_homo_f32_i32);
        dict["gpu_atomic_sum_homo_f32_i64"] = EncapsulateFunction(gpu_atomic_sum_homo_f32_i64);
        dict["gpu_atomic_sum_homo_f64_i32"] = EncapsulateFunction(gpu_atomic_sum_homo_f64_i32);
        dict["gpu_atomic_sum_homo_f64_i64"] = EncapsulateFunction(gpu_atomic_sum_homo_f64_i64);
        // heterogeneous atomic_sum
        dict["gpu_atomic_sum_heter_f32_i32"] = EncapsulateFunction(gpu_atomic_sum_heter_f32_i32);
        dict["gpu_atomic_sum_heter_f32_i64"] = EncapsulateFunction(gpu_atomic_sum_heter_f32_i64);
        dict["gpu_atomic_sum_heter_f64_i32"] = EncapsulateFunction(gpu_atomic_sum_heter_f64_i32);
        dict["gpu_atomic_sum_heter_f64_i64"] = EncapsulateFunction(gpu_atomic_sum_heter_f64_i64);

        // homogeneous atomic_prod
        dict["gpu_atomic_prod_homo_f32_i32"] = EncapsulateFunction(gpu_atomic_prod_homo_f32_i32);
        dict["gpu_atomic_prod_homo_f32_i64"] = EncapsulateFunction(gpu_atomic_prod_homo_f32_i64);
        dict["gpu_atomic_prod_homo_f64_i32"] = EncapsulateFunction(gpu_atomic_prod_homo_f64_i32);
        dict["gpu_atomic_prod_homo_f64_i64"] = EncapsulateFunction(gpu_atomic_prod_homo_f64_i64);
        // heterogeneous atomic_prod
        dict["gpu_atomic_prod_heter_f32_i32"] = EncapsulateFunction(gpu_atomic_prod_heter_f32_i32);
        dict["gpu_atomic_prod_heter_f32_i64"] = EncapsulateFunction(gpu_atomic_prod_heter_f32_i64);
        dict["gpu_atomic_prod_heter_f64_i32"] = EncapsulateFunction(gpu_atomic_prod_heter_f64_i32);
        dict["gpu_atomic_prod_heter_f64_i64"] = EncapsulateFunction(gpu_atomic_prod_heter_f64_i64);

        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m
    ) {
    m.def("registrations", &Registrations);
    m.def("build_event_sum_descriptor", &build_event_sum_descriptor);
    m.def("build_event_sum2_descriptor", &build_event_sum2_descriptor);
    m.def("build_atomic_sum_descriptor", &build_atomic_sum_descriptor);
    m.def("build_atomic_prod_descriptor", &build_atomic_prod_descriptor);
}
}  // namespace
