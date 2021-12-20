#ifndef _BRAINPY_EVENT_ADD_H_
#define _BRAINPY_EVENT_ADD_H_

#include <cstdint>
#include <cmath>

namespace brainpy_lib {
    void cpu_event_add_f32_i32(void *out, const void **in);
    void cpu_event_add_f32_i64(void *out, const void **in);
    void cpu_event_add_f64_i32(void *out, const void **in);
    void cpu_event_add_f64_i64(void *out, const void **in);

    void cpu_event_add_heter_f32_i32(void *out, const void **in);
    void cpu_event_add_heter_f32_i64(void *out, const void **in);
    void cpu_event_add_heter_f64_i32(void *out, const void **in);
    void cpu_event_add_heter_f64_i64(void *out, const void **in);

    void cpu_event_add2_f32_i32(void *out, const void **in);
    void cpu_event_add2_f32_i64(void *out, const void **in);
    void cpu_event_add2_f64_i32(void *out, const void **in);
    void cpu_event_add2_f64_i64(void *out, const void **in);
}

#endif