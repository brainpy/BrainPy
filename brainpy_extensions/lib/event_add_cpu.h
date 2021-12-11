#ifndef _BRAINPY_EVENT_ADD_H_
#define _BRAINPY_EVENT_ADD_H_

#include <cstdint>

namespace brainpy_lib {
    void cpu_event_add(void *out, const void **in);
    void cpu_event_add_v2(void *out, const void **in);
}

#endif