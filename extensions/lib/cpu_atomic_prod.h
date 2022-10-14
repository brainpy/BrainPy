#ifndef _BRAINPYLIB_ATOMIC_prod_H_
#define _BRAINPYLIB_ATOMIC_prod_H_

#include <cstdint>
#include <cstring>
#include <cmath>

namespace brainpy_lib {
    void cpu_coo_atomic_prod_heter_f32_i32(void *out, const void **in);
    void cpu_coo_atomic_prod_heter_f32_i64(void *out, const void **in);
    void cpu_coo_atomic_prod_heter_f64_i32(void *out, const void **in);
    void cpu_coo_atomic_prod_heter_f64_i64(void *out, const void **in);
    
    void cpu_coo_atomic_prod_homo_f32_i32(void *out, const void **in);
    void cpu_coo_atomic_prod_homo_f32_i64(void *out, const void **in);
    void cpu_coo_atomic_prod_homo_f64_i32(void *out, const void **in);
    void cpu_coo_atomic_prod_homo_f64_i64(void *out, const void **in);
}

#endif