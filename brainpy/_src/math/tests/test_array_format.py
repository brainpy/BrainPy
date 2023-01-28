

import brainpy.math as bm


def test_format():
  print(bm.ones((5)))
  print(bm.Variable(bm.ones((5))))
  print(bm.VariableView(bm.Variable(bm.ones((5))), bm.asarray([1, 2, 3])))

  print(bm.ones((3, 4)))
  print(bm.Variable(bm.ones((3, 4))))
  print(bm.VariableView(bm.Variable(bm.ones((3, 4))), bm.asarray([1, 2])))

