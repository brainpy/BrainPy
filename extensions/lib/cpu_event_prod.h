#ifndef _BRAINPYLIB_EVENT_PROD_H_
#define _BRAINPYLIB_EVENT_PROD_H_

#include <cstdint>
#include <cstring>
#include <cmath>

namespace brainpy_lib {
    // "values" is homogeneous
    void cpu_csr_event_prod_homo_f32_i32(void *out, const void **in);
    void cpu_csr_event_prod_homo_f32_i64(void *out, const void **in);
    void cpu_csr_event_prod_homo_f64_i32(void *out, const void **in);
    void cpu_csr_event_prod_homo_f64_i64(void *out, const void **in);
    // "values" is heterogeneous
    void cpu_csr_event_prod_heter_f32_i32(void *out, const void **in);
    void cpu_csr_event_prod_heter_f32_i64(void *out, const void **in);
    void cpu_csr_event_prod_heter_f64_i32(void *out, const void **in);
    void cpu_csr_event_prod_heter_f64_i64(void *out, const void **in);
}

#endif