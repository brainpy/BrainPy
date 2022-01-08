#include "event_sum_cpu.h"

namespace brainpy_lib {
namespace{
    template <typename F, typename I>
    void cpu_event_sum_homo(void *out, const void **in) {
      // Parse the inputs
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F *value = reinterpret_cast<const F *>(in[5]);
      const F value = value[0];

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(result[0]) * post_size);
      for (std::uint32_t i=0; i<pre_size; ++i) {
        if (events[i]){
          for (I j=indptr[i]; j<indptr[i+1]; ++j) {
            result[indices[j]] += value;
          }
        }
      }
    }

    template <typename F, typename I>
    void cpu_event_sum_heter(void *out, const void **in) {
      // Parse the inputs
      const std::uint32_t pre_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const I *indices = reinterpret_cast<const I *>(in[3]);
      const I *indptr = reinterpret_cast<const I *>(in[4]);
      const F *values = reinterpret_cast<const F *>(in[5]);

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(result[0]) * post_size);
      for (std::uint32_t i = 0; i < pre_size; ++i) {
        if (events[i]){
          for (I j = indptr[i]; j < indptr[i+1]; ++j) {
            result[indices[j]] += values[j];
          }
        }
      }
    }

    template <typename F, typename I>
    void cpu_event_sum2_homo(void *out, const void **in) {
      // The inputs
      const std::uint32_t conn_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const    I *pre_ids = reinterpret_cast<const   I *>(in[3]);
      const    I *post_ids = reinterpret_cast<const  I *>(in[4]);
      const    F *value = reinterpret_cast<const F *>(in[5]);
      const    F  value = value[0];

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(result[0]) * post_size);
      for (std::uint32_t i=0; i<conn_size; ++i) {
        if (events[pre_ids[i]]){
          result[post_ids[i]] += value;
        }
      }
    }

    template <typename F, typename I>
    void cpu_event_sum2_heter(void *out, const void **in) {
      // The inputs
      const std::uint32_t conn_size = *reinterpret_cast<const std::uint32_t *>(in[0]);
      const std::uint32_t post_size = *reinterpret_cast<const std::uint32_t *>(in[1]);
      const bool *events = reinterpret_cast<const bool *>(in[2]);
      const    I *pre_ids = reinterpret_cast<const   I *>(in[3]);
      const    I *post_ids = reinterpret_cast<const  I *>(in[4]);
      const    F *values = reinterpret_cast<const F *>(in[5]);

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      memset(&result[0], 0, sizeof(result[0]) * post_size);
      for (std::uint32_t i=0; i<conn_size; ++i) {
        if (events[pre_ids[i]]){
          const I post_i = post_ids[i];
          result[post_i] += values[post_i];
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

void cpu_event_sum2_homo_f32_i32(void *out, const void **in){cpu_event_sum2_homo<float, std::uint32_t>(out, in);}
void cpu_event_sum2_homo_f32_i64(void *out, const void **in){cpu_event_sum2_homo<float, std::uint64_t>(out, in);}
void cpu_event_sum2_homo_f64_i32(void *out, const void **in){cpu_event_sum2_homo<double, std::uint32_t>(out, in);}
void cpu_event_sum2_homo_f64_i64(void *out, const void **in){cpu_event_sum2_homo<double, std::uint64_t>(out, in);}

void cpu_event_sum2_heter_f32_i32(void *out, const void **in){cpu_event_sum2_heter<float, std::uint32_t>(out, in);}
void cpu_event_sum2_heter_f32_i64(void *out, const void **in){cpu_event_sum2_heter<float, std::uint64_t>(out, in);}
void cpu_event_sum2_heter_f64_i32(void *out, const void **in){cpu_event_sum2_heter<double, std::uint32_t>(out, in);}
void cpu_event_sum2_heter_f64_i64(void *out, const void **in){cpu_event_sum2_heter<double, std::uint64_t>(out, in);}
}
