# -*- coding: utf-8 -*-

from brainpy._src.math.operators.op_registers import (
  XLACustomOp as XLACustomOp,
)

from brainpy._src.math.operators.pre_syn_post import (
  pre2post_sum as pre2post_sum,
  pre2post_prod as pre2post_prod,
  pre2post_max as pre2post_max,
  pre2post_min as pre2post_min,
  pre2post_mean as pre2post_mean,
  pre2post_event_sum as pre2post_event_sum,
  pre2post_coo_event_sum as pre2post_coo_event_sum,
  pre2post_event_prod as pre2post_event_prod,
  pre2syn as pre2syn,
  syn2post_sum as syn2post_sum,
  syn2post as syn2post,
  syn2post_prod as syn2post_prod,
  syn2post_max as syn2post_max,
  syn2post_min as syn2post_min,
  syn2post_mean as syn2post_mean,
  syn2post_softmax as syn2post_softmax,
)

from brainpy._src.math.operators.sparse_matmul import (
  sparse_matmul as sparse_matmul,
  csr_matvec as csr_matvec,
)

from brainpy._src.math.operators.event_ops import (
  event_csr_matvec as event_csr_matvec,
  event_info as event_info,
)
from brainpy._src.math.operators.sparse_ops import (
  csr_matvec as csr_matvec,
  cusparse_csr_matvec as cusparse_csr_matvec,
  cusparse_coo_matvec as cusparse_coo_matvec,
)

from brainpy._src.math.operators.jitconn_ops import (
  matvec_prob_conn_homo_weight as matvec_prob_conn_homo_weight,
  matvec_prob_conn_uniform_weight as matvec_prob_conn_uniform_weight,
  matvec_prob_conn_normal_weight as matvec_prob_conn_normal_weight,
  event_matvec_prob_conn_homo_weight as event_matvec_prob_conn_homo_weight,
  event_matvec_prob_conn_uniform_weight as event_matvec_prob_conn_uniform_weight,
  event_matvec_prob_conn_normal_weight as event_matvec_prob_conn_normal_weight,
  matmat_prob_conn_uniform_weight as matmat_prob_conn_uniform_weight,
  matmat_prob_conn_normal_weight as matmat_prob_conn_normal_weight
)

from brainpy._src.math.operators.compat import (
  coo_atomic_sum as coo_atomic_sum,
  coo_atomic_prod as coo_atomic_prod,
  csr_event_sum as csr_event_sum,
  coo_event_sum as coo_event_sum,
  csr_event_prod as csr_event_prod,
)