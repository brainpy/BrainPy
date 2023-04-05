# -*- coding: utf-8 -*-


from brainpy._src.math.operators.event_ops import (
  event_csr_matvec, 
  event_info
)
from brainpy._src.math.operators.jitconn_ops import (
  event_matvec_prob_conn_homo_weight,
  event_matvec_prob_conn_uniform_weight,
  event_matvec_prob_conn_normal_weight,

  matmat_prob_conn_homo_weight,
  matmat_prob_conn_uniform_weight,
  matmat_prob_conn_normal_weight,

  matvec_prob_conn_homo_weight,
  matvec_prob_conn_uniform_weight,
  matvec_prob_conn_normal_weight,
)
from brainpy._src.math.operators.op_registers import (
  XLACustomOp,
  compile_cpu_signature_with_numba,
)
from brainpy._src.math.operators.pre_syn_post import (
  pre2post_sum,
  pre2post_prod,
  pre2post_max,
  pre2post_min,
  pre2post_mean,

  pre2post_event_sum,
  pre2post_coo_event_sum,
  pre2post_event_prod,

  pre2syn,

  syn2post_sum, syn2post,
  syn2post_prod,
  syn2post_max,
  syn2post_min,
  syn2post_mean,
  syn2post_softmax,
)
from brainpy._src.math.operators.sparse_ops import (
  cusparse_csr_matvec,
  cusparse_coo_matvec,
  csr_matvec,
  sparse_matmul,
  coo_to_csr,
  csr_to_coo,
  csr_to_dense
)

