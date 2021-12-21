#ifndef _BRAINPY_event_sum_H_
#define _BRAINPY_event_sum_H_

#include <cstdint>
#include <string.h>
#include <cmath>

namespace brainpy_lib {
    void cpu_event_sum_homo_f32_i32(void *out, const void **in);
    void cpu_event_sum_homo_f32_i64(void *out, const void **in);
    void cpu_event_sum_homo_f64_i32(void *out, const void **in);
    void cpu_event_sum_homo_f64_i64(void *out, const void **in);

    void cpu_event_sum_heter_f32_i32(void *out, const void **in);
    void cpu_event_sum_heter_f32_i64(void *out, const void **in);
    void cpu_event_sum_heter_f64_i32(void *out, const void **in);
    void cpu_event_sum_heter_f64_i64(void *out, const void **in);

    void cpu_event_sum2_f32_i32(void *out, const void **in);
    void cpu_event_sum2_f32_i64(void *out, const void **in);
    void cpu_event_sum2_f64_i32(void *out, const void **in);
    void cpu_event_sum2_f64_i64(void *out, const void **in);
}

#endif