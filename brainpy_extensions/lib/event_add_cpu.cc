#include "event_add_cpu.h"

namespace brainpy_lib {
    template <typename T>
    void cpu_event_add(void *out, const void **in) {
      // Parse the inputs
      const std::int64_t pre_size = *reinterpret_cast<const std::int64_t *>(in[0]);
      const bool *events = reinterpret_cast<const bool *>(in[1]);
      const std::int64_t *indices = reinterpret_cast<const std::int64_t *>(in[2]);
      const std::int64_t *indptr = reinterpret_cast<const std::int64_t *>(in[3]);
      const T value = *reinterpret_cast<const T *>(in[4]);

      // The output
      T *result = reinterpret_cast<T *>(out);

      // algorithm
      for (std::int64_t i = 0; i < pre_size; ++i) {
        if (events[i]){
          for (std::int64_t j = indptr[i]; j < indptr[i+1]; j++) {
            result[indices[j]] += value;
          }
        }
      }
    }

    template <typename T>
    void cpu_event_add_v2(void *out, const void **in) {
      // The inputs
      const bool *events = reinterpret_cast<const bool *>(in[0]);
      const std::int64_t *pre_ids = reinterpret_cast<const std::int64_t *>(in[1]);
      const std::int64_t *post_ids = reinterpret_cast<const std::int64_t *>(in[2]);
      const std::int64_t conn_size = *reinterpret_cast<const std::int64_t *>(in[3]);
      const T value = *reinterpret_cast<const T *>(in[4]);

      // The output
      T *result = reinterpret_cast<T *>(out);

      // algorithm
      for (std::int64_t i=0; i<conn_size; ++i) {
        if (events[pre_ids[i]]){
          result[post_ids[i]] += value;
        }
      }
    }
}
