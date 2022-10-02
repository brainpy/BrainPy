#include "event_sum_cpu.h"

namespace brainpy_lib {
namespace{
    template <typename F, typename I>
    void cpu_event_sum_homo(void *out, const void **in) {
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F weight = *reinterpret_cast<const F *>(in[5]);
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(F) * post_size);
      for (std::uint32_t i=0; i<pre_size; ++i) {
        if (events[i]){
          for (I j=indptr[i]; j<indptr[i+1]; ++j) {
            result[indices[j]] += weight;
          }
        }
      }
    }

    // TODO:: batch version of "event_sum_homo" CPU operator
    template <typename F, typename I>
    void cpu_event_sum_batch_homo(void *out, const void **in) {
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F weight = *reinterpret_cast<const F *>(in[5]);
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(F) * post_size);
      for (std::uint32_t i=0; i<pre_size; ++i) {
        if (events[i]){
          for (I j=indptr[i]; j<indptr[i+1]; ++j) {
            result[indices[j]] += weight;
          }
        }
      }
    }

    template <typename F, typename I>
    void cpu_event_sum_heter(void *out, const void **in) {
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F *values = reinterpret_cast<const F *>(in[5]);
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(F) * post_size);
      for (std::uint32_t i = 0; i < pre_size; ++i) {
        if (events[i]){
          for (I j = indptr[i]; j < indptr[i+1]; ++j) {
            result[indices[j]] += values[j];
          }
        }
      }
    }


    // TODO:: batch version of "event_sum_heter" CPU operator
    template <typename F, typename I>
    void cpu_event_sum_batch_heter(void *out, const void **in) {
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F *values = reinterpret_cast<const F *>(in[5]);
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(F) * post_size);
      for (std::uint32_t i = 0; i < pre_size; ++i) {
        if (events[i]){
          for (I j = indptr[i]; j < indptr[i+1]; ++j) {
            result[indices[j]] += values[j];
          }
        }
      }
    }


}

void cpu_event_sum_homo_f32_i32(void *out, const void **in){cpu_event_sum_homo<float, std::uint32_t>(out, in);}
void cpu_event_sum_homo_f32_i64(void *out, const void **in){cpu_event_sum_homo<float, std::uint64_t>(out, in);}
void cpu_event_sum_homo_f64_i32(void *out, const void **in){cpu_event_sum_homo<double, std::uint32_t>(out, in);}
void cpu_event_sum_homo_f64_i64(void *out, const void **in){cpu_event_sum_homo<double, std::uint64_t>(out, in);}

void cpu_event_sum_heter_f32_i32(void *out, const void **in){cpu_event_sum_heter<float, std::uint32_t>(out, in);}
void cpu_event_sum_heter_f32_i64(void *out, const void **in){cpu_event_sum_heter<float, std::uint64_t>(out, in);}
void cpu_event_sum_heter_f64_i32(void *out, const void **in){cpu_event_sum_heter<double, std::uint32_t>(out, in);}
void cpu_event_sum_heter_f64_i64(void *out, const void **in){cpu_event_sum_heter<double, std::uint64_t>(out, in);}
}
