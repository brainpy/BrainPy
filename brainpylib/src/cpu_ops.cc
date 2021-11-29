// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"

using namespace brainpy_extension;

namespace {

    template <typename T>
    void cpu_event_add(void *out, const void **in) {
      // Parse the inputs
      const std::int64_t pre_size = *reinterpret_cast<const std::int64_t *>(in[0]);
      const std::int64_t post_size = *reinterpret_cast<const std::int64_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const bool *conn = reinterpret_cast<const bool *>(in[3]);
      const T value = *reinterpret_cast<const T *>(in[4]);

      // The output
      T *result = reinterpret_cast<T *>(out);

      // algorithm
      for (std::int64_t i = 0; i < pre_size; ++i) {
        if (events[i]){
          for (std::int64_t j = 0; j < post_size; ++j) {
            if (conn[i * post_size + j]){
              result[j] += value;
            }
          }
        }
      }
    }

    template <typename T>
    void cpu_event_slice_add(void *out, const void **in) {
      // Parse the inputs
      const std::int64_t pre_size = *reinterpret_cast<const std::int64_t *>(in[0]);
      const bool *events = reinterpret_cast<const bool *>(in[1]);
      const int *post_ids = reinterpret_cast<const int *>(in[2]);
      const int *pre_slice = reinterpret_cast<const int *>(in[3]);
      const T value = *reinterpret_cast<const T *>(in[4]);

      // The output
      T *result = reinterpret_cast<T *>(out);

      // algorithm
      for (int i = 0; i < pre_size; ++i) {
        if (events[i]){
          int stop = pre_slice[i * 2 + 1];
          for (int j = pre_slice[i * 2]; j < stop; j++) {
            result[post_ids[j]] += value;
          }
        }
      }
    }

    pybind11::dict Registrations() {
      pybind11::dict dict;
      dict["cpu_event_add_f32"] = EncapsulateFunction(cpu_event_add<float>);
      dict["cpu_event_add_f64"] = EncapsulateFunction(cpu_event_add<double>);
      dict["cpu_event_slice_add_f32"] = EncapsulateFunction(cpu_event_slice_add<float>);
      dict["cpu_event_slice_add_f64"] = EncapsulateFunction(cpu_event_slice_add<double>);
      return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace
