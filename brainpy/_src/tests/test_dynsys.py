
import brainpy as bp


def test1():
  class A(bp.DynamicalSystem):
    def update(self, x=None):
      # print(tdi)
      print(x)

  A()({}, 10.)


def test2():
  class B(bp.DynamicalSystem):
    def update(self, tdi, x=None):
      print(tdi)
      print(x)

  B()({}, 10.)
  B()(10.)


def test3():
  class A(bp.DynamicalSystem):
    def update(self, x=None):
      # print(tdi)
      print('A:', x)

  class B(A):
    def update(self, tdi, x=None):
      print('B:', tdi, x)
      super().update(x)

  B()(dict(), 1.)
  B()(1.)




