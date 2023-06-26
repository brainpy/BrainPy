# -*- coding: utf-8 -*-

from brainpy._src.connect.base import (
  Connector as Connector,
  TwoEndConnector as TwoEndConnector,
  OneEndConnector as OneEndConnector,
  mat2coo as mat2coo,
  mat2csc as mat2csc,
  mat2csr as mat2csr,
  csr2csc as csr2csc,
  csr2mat as csr2mat,
  csr2coo as csr2coo,
  coo2csr as coo2csr,
  coo2csc as coo2csc,
  coo2mat as coo2mat,
  visualizeMat as visualizeMat,

  CONN_MAT,
  PRE_IDS, POST_IDS,
  PRE2POST, POST2PRE,
  PRE2SYN, POST2SYN,
  PRE_SLICE, POST_SLICE,
  COO, CSR, CSC
)

from brainpy._src.connect.custom_conn import (
  MatConn as MatConn,
  IJConn as IJConn,
  CSRConn as CSRConn,
  SparseMatConn as SparseMatConn,
)

from brainpy._src.connect.random_conn import (
  FixedProb as FixedProb,
  FixedPreNum as FixedPreNum,
  FixedPostNum as FixedPostNum,
  FixedTotalNum as FixedTotalNum,
  GaussianProb as GaussianProb,
  ProbDist as ProbDist,
  SmallWorld as SmallWorld,
  ScaleFreeBA as ScaleFreeBA,
  ScaleFreeBADual as ScaleFreeBADual,
  PowerLaw as PowerLaw,
)


from brainpy._src.connect.regular_conn import (
  One2One as One2One,
  one2one as one2one,
  All2All as All2All,
  all2all as all2all,
  GridFour as GridFour,
  grid_four as grid_four,
  GridEight as GridEight,
  grid_eight as grid_eight,
  GridN as GridN,
)
