// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include "event_add_cpu.h"

using namespace kepler_jax;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;
      dict["cpu_event_add_f32"] = EncapsulateFunction(cpu_event_add<float>);
      dict["cpu_event_add_f64"] = EncapsulateFunction(cpu_event_add<double>);
      dict["cpu_event_add_v2_f32"] = EncapsulateFunction(cpu_event_add_v2<float>);
      dict["cpu_event_add_v2_f64"] = EncapsulateFunction(cpu_event_add_v2<double>);
      return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
