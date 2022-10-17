// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include "pybind11_kernel_helpers.h"
#include "gpu_event_sum.h"
#include "gpu_atomic_sum.h"
#include "gpu_atomic_prod.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
        pybind11::dict dict;

        // homogeneous csr event_sum 
        dict["gpu_csr_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_csr_event_sum_homo_f32_i32);
        dict["gpu_csr_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_csr_event_sum_homo_f32_i64);
        dict["gpu_csr_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_csr_event_sum_homo_f64_i32);
        dict["gpu_csr_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_csr_event_sum_homo_f64_i64);
        // heterogeneous csr event_sum
        dict["gpu_csr_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_csr_event_sum_heter_f32_i32);
        dict["gpu_csr_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_csr_event_sum_heter_f32_i64);
        dict["gpu_csr_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_csr_event_sum_heter_f64_i32);
        dict["gpu_csr_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_csr_event_sum_heter_f64_i64);

        // homogeneous coo event_sum
        dict["gpu_coo_event_sum_homo_f32_i32"] = EncapsulateFunction(gpu_coo_event_sum_homo_f32_i32);
        dict["gpu_coo_event_sum_homo_f32_i64"] = EncapsulateFunction(gpu_coo_event_sum_homo_f32_i64);
        dict["gpu_coo_event_sum_homo_f64_i32"] = EncapsulateFunction(gpu_coo_event_sum_homo_f64_i32);
        dict["gpu_coo_event_sum_homo_f64_i64"] = EncapsulateFunction(gpu_coo_event_sum_homo_f64_i64);
        // heterogeneous coo event_sum
        dict["gpu_coo_event_sum_heter_f32_i32"] = EncapsulateFunction(gpu_coo_event_sum_heter_f32_i32);
        dict["gpu_coo_event_sum_heter_f32_i64"] = EncapsulateFunction(gpu_coo_event_sum_heter_f32_i64);
        dict["gpu_coo_event_sum_heter_f64_i32"] = EncapsulateFunction(gpu_coo_event_sum_heter_f64_i32);
        dict["gpu_coo_event_sum_heter_f64_i64"] = EncapsulateFunction(gpu_coo_event_sum_heter_f64_i64);

        // homogeneous atomic_sum
        dict["gpu_coo_atomic_sum_homo_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f32_i32);
        dict["gpu_coo_atomic_sum_homo_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f32_i64);
        dict["gpu_coo_atomic_sum_homo_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f64_i32);
        dict["gpu_coo_atomic_sum_homo_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_homo_f64_i64);
        // heterogeneous atomic_sum
        dict["gpu_coo_atomic_sum_heter_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f32_i32);
        dict["gpu_coo_atomic_sum_heter_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f32_i64);
        dict["gpu_coo_atomic_sum_heter_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f64_i32);
        dict["gpu_coo_atomic_sum_heter_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_sum_heter_f64_i64);

        // homogeneous atomic_prod
        dict["gpu_coo_atomic_prod_homo_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f32_i32);
        dict["gpu_coo_atomic_prod_homo_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f32_i64);
        dict["gpu_coo_atomic_prod_homo_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f64_i32);
        dict["gpu_coo_atomic_prod_homo_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_homo_f64_i64);
        // heterogeneous atomic_prod
        dict["gpu_coo_atomic_prod_heter_f32_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f32_i32);
        dict["gpu_coo_atomic_prod_heter_f32_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f32_i64);
        dict["gpu_coo_atomic_prod_heter_f64_i32"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f64_i32);
        dict["gpu_coo_atomic_prod_heter_f64_i64"] = EncapsulateFunction(gpu_coo_atomic_prod_heter_f64_i64);

        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m
    ) {
    m.def("registrations", &Registrations);
    m.def("build_csr_event_sum_descriptor", &build_csr_event_sum_descriptor);
    m.def("build_coo_event_sum_descriptor", &build_coo_event_sum_descriptor);
    m.def("build_coo_atomic_sum_descriptor", &build_coo_atomic_sum_descriptor);
    m.def("build_coo_atomic_prod_descriptor", &build_coo_atomic_prod_descriptor);
}
}  // namespace
