#include "cpu_atomic_prod.h"

namespace brainpy_lib {
namespace{
    template <typename F, typename I>
    void cpu_coo_atomic_prod_heter(void *out, const void **in) {
      // The inputs
      const F *values = reinterpret_cast<const F *>(in[0]);
      const I *pre_ids = reinterpret_cast<const I *>(in[1]);
      const I *post_ids = reinterpret_cast<const I *>(in[2]);
      const std::uint32_t conn_size = *reinterpret_cast<const std::uint32_t *>(in[3]);
      const std::uint32_t out_size = *reinterpret_cast<const std::uint32_t *>(in[4]);

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      for(std::uint32_t ii = 0; ii < out_size; ++ii)
        result[ii] = 1.;
      for (std::uint32_t i=0; i<conn_size; ++i) {
        result[post_ids[i]] *= values[pre_ids[i]];
      }
    }
    
    template <typename F, typename I>
    void cpu_coo_atomic_prod_homo(void *out, const void **in) {
      // The inputs
      const F *values = reinterpret_cast<const F *>(in[0]);  // scalar as a vector
      const F value = values[0];
      const I *post_ids = reinterpret_cast<const I *>(in[1]);
      const std::uint32_t conn_size = *reinterpret_cast<const std::uint32_t *>(in[2]);
      const std::uint32_t out_size = *reinterpret_cast<const std::uint32_t *>(in[3]);

      // The output
      F *result = reinterpret_cast<F *>(out);

      // algorithm
      for(std::uint32_t ii = 0; ii < out_size; ++ii)
        result[ii] = 1.;
      for (std::uint32_t i=0; i<conn_size; ++i) {
        result[post_ids[i]] *= value;
      }
    }
}

void cpu_coo_atomic_prod_heter_f32_i32(void *out, const void **in){
    cpu_coo_atomic_prod_heter<float, std::uint32_t>(out, in);}
void cpu_coo_atomic_prod_heter_f32_i64(void *out, const void **in){
    cpu_coo_atomic_prod_heter<float, std::uint64_t>(out, in);}
void cpu_coo_atomic_prod_heter_f64_i32(void *out, const void **in){
    cpu_coo_atomic_prod_heter<double, std::uint32_t>(out, in);}
void cpu_coo_atomic_prod_heter_f64_i64(void *out, const void **in){
    cpu_coo_atomic_prod_heter<double, std::uint64_t>(out, in);}

void cpu_coo_atomic_prod_homo_f32_i32(void *out, const void **in){
    cpu_coo_atomic_prod_homo<float, std::uint32_t>(out, in);}
void cpu_coo_atomic_prod_homo_f32_i64(void *out, const void **in){
    cpu_coo_atomic_prod_homo<float, std::uint64_t>(out, in);}
void cpu_coo_atomic_prod_homo_f64_i32(void *out, const void **in){
    cpu_coo_atomic_prod_homo<double, std::uint32_t>(out, in);}
void cpu_coo_atomic_prod_homo_f64_i64(void *out, const void **in){
    cpu_coo_atomic_prod_homo<double, std::uint64_t>(out, in);}

}
