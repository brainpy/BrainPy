# -*- coding: utf-8 -*-


from brainpy.simulation import DynamicSystem


class TestDynamicSystem(DynamicSystem):
    target_backend = 'numpy'


def try1():
    ch_d1 = TestDynamicSystem()
    ch_d2 = TestDynamicSystem()
    parent = TestDynamicSystem()
    print(hash(ch_d1))
    print(hash(ch_d2))
    print(hash(parent))

    parent.ch1 = ch_d1
    parent.ch2 = ch_d2
    print(parent.contained_members)
    print(parent.children_nodes)


if __name__ == '__main__':
    try1()
