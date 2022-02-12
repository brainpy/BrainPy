// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "event_sum_cpu.h"
#include "event_prod_cpu.h"
#include "atomic_sum_cpu.h"
#include "atomic_prod_cpu.h"
#include "pybind11_kernel_helpers.h"

using namespace brainpy_lib;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;

      // event_sum for homogeneous value
      dict["cpu_event_sum_homo_f32_i32"] = EncapsulateFunction(cpu_event_sum_homo_f32_i32);
      dict["cpu_event_sum_homo_f32_i64"] = EncapsulateFunction(cpu_event_sum_homo_f32_i64);
      dict["cpu_event_sum_homo_f64_i32"] = EncapsulateFunction(cpu_event_sum_homo_f64_i32);
      dict["cpu_event_sum_homo_f64_i64"] = EncapsulateFunction(cpu_event_sum_homo_f64_i64);
      // event_sum for heterogeneous values
      dict["cpu_event_sum_heter_f32_i32"] = EncapsulateFunction(cpu_event_sum_heter_f32_i32);
      dict["cpu_event_sum_heter_f32_i64"] = EncapsulateFunction(cpu_event_sum_heter_f32_i64);
      dict["cpu_event_sum_heter_f64_i32"] = EncapsulateFunction(cpu_event_sum_heter_f64_i32);
      dict["cpu_event_sum_heter_f64_i64"] = EncapsulateFunction(cpu_event_sum_heter_f64_i64);

      // event_prod for homogeneous value
      dict["cpu_event_prod_homo_f32_i32"] = EncapsulateFunction(cpu_event_prod_homo_f32_i32);
      dict["cpu_event_prod_homo_f32_i64"] = EncapsulateFunction(cpu_event_prod_homo_f32_i64);
      dict["cpu_event_prod_homo_f64_i32"] = EncapsulateFunction(cpu_event_prod_homo_f64_i32);
      dict["cpu_event_prod_homo_f64_i64"] = EncapsulateFunction(cpu_event_prod_homo_f64_i64);
      // event_prod for heterogeneous values
      dict["cpu_event_prod_heter_f32_i32"] = EncapsulateFunction(cpu_event_prod_heter_f32_i32);
      dict["cpu_event_prod_heter_f32_i64"] = EncapsulateFunction(cpu_event_prod_heter_f32_i64);
      dict["cpu_event_prod_heter_f64_i32"] = EncapsulateFunction(cpu_event_prod_heter_f64_i32);
      dict["cpu_event_prod_heter_f64_i64"] = EncapsulateFunction(cpu_event_prod_heter_f64_i64);

      // atomic_sum for heterogeneous values
      dict["cpu_atomic_sum_heter_f32_i32"] = EncapsulateFunction(cpu_atomic_sum_heter_f32_i32);
      dict["cpu_atomic_sum_heter_f32_i64"] = EncapsulateFunction(cpu_atomic_sum_heter_f32_i64);
      dict["cpu_atomic_sum_heter_f64_i32"] = EncapsulateFunction(cpu_atomic_sum_heter_f64_i32);
      dict["cpu_atomic_sum_heter_f64_i64"] = EncapsulateFunction(cpu_atomic_sum_heter_f64_i64);
      // atomic_sum for homogeneous value
      dict["cpu_atomic_sum_homo_f32_i32"] = EncapsulateFunction(cpu_atomic_sum_homo_f32_i32);
      dict["cpu_atomic_sum_homo_f32_i64"] = EncapsulateFunction(cpu_atomic_sum_homo_f32_i64);
      dict["cpu_atomic_sum_homo_f64_i32"] = EncapsulateFunction(cpu_atomic_sum_homo_f64_i32);
      dict["cpu_atomic_sum_homo_f64_i64"] = EncapsulateFunction(cpu_atomic_sum_homo_f64_i64);
      
      // atomic_prod for heterogeneous values
      dict["cpu_atomic_prod_heter_f32_i32"] = EncapsulateFunction(cpu_atomic_prod_heter_f32_i32);
      dict["cpu_atomic_prod_heter_f32_i64"] = EncapsulateFunction(cpu_atomic_prod_heter_f32_i64);
      dict["cpu_atomic_prod_heter_f64_i32"] = EncapsulateFunction(cpu_atomic_prod_heter_f64_i32);
      dict["cpu_atomic_prod_heter_f64_i64"] = EncapsulateFunction(cpu_atomic_prod_heter_f64_i64);
      // atomic_prod for homogeneous value
      dict["cpu_atomic_prod_homo_f32_i32"] = EncapsulateFunction(cpu_atomic_prod_homo_f32_i32);
      dict["cpu_atomic_prod_homo_f32_i64"] = EncapsulateFunction(cpu_atomic_prod_homo_f32_i64);
      dict["cpu_atomic_prod_homo_f64_i32"] = EncapsulateFunction(cpu_atomic_prod_homo_f64_i32);
      dict["cpu_atomic_prod_homo_f64_i64"] = EncapsulateFunction(cpu_atomic_prod_homo_f64_i64);
      
      return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) {
        m.def("registrations", &Registrations);
    }

}  // namespace
