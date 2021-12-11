#ifndef _BRAINPY_EVENT_ADD_H_
#define _BRAINPY_EVENT_ADD_H_

#include <cstdint>
#include <cmath>

namespace brainpy_lib {
    void cpu_event_add_f32(void *out, const void **in);
    void cpu_event_add_f64(void *out, const void **in);
    void cpu_event_add_v2_f32(void *out, const void **in);
    void cpu_event_add_v2_f64(void *out, const void **in);
}

#endif