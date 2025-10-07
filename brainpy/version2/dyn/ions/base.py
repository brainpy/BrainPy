# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Union, Optional, Dict, Sequence, Callable

import brainpy.version2.math as bm
from brainpy.mixin import Container, TreeNode
from brainpy.version2.dyn.base import IonChaDyn
from brainpy.version2.dyn.neurons.hh import HHTypedNeuron
from brainpy.version2.types import Shape
from brainstate.mixin import _JointGenericAlias

__all__ = [
    'MixIons',
    'mix_ions',
    'Ion',
]


class MixIons(IonChaDyn, Container, TreeNode):
    """Mixing Ions.

    Args:
      ions: Instances of ions. This option defines the master types of all children objects.
      channels: Instance of channels.
    """
    master_type = HHTypedNeuron

    def __init__(self, *ions, **channels):
        # TODO: check "ions" should be independent from each other
        assert isinstance(ions, (tuple, list)), f'{self.__class__.__name__} requires at least two ions. '
        assert len(ions) >= 2, f'{self.__class__.__name__} requires at least two ions. '
        assert all([isinstance(cls, Ion) for cls in ions]), f'Must be a sequence of Ion. But got {ions}.'
        super().__init__(size=ions[0].size, keep_size=ions[0].keep_size, sharding=ions[0].sharding)

        # Attribute of "Container"
        self.children = bm.node_dict()

        self.ions: Sequence['Ion'] = tuple(ions)
        self._ion_classes = tuple([type(ion) for ion in self.ions])
        for k, v in channels.items():
            self.add_elem(k=v)

    def update(self, V):
        nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())
        self.check_hierarchies(self._ion_classes, *nodes)
        for node in nodes:
            infos = tuple([self._get_imp(root).pack_info() for root in node.master_type.__args__])
            node.update(V, *infos)

    def current(self, V):
        """Generate ion channel current.

        Args:
          V: The membrane potential.

        Returns:
          Current.
        """
        nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())
        self.check_hierarchies(self._ion_classes, *nodes)

        if len(nodes) == 0:
            return 0.
        else:
            current = 0.
            for node in nodes:
                infos = tuple([self._get_imp(root).pack_info() for root in node.master_type.__args__])
                current = current + node.current(V, *infos)
            return current

    def reset_state(self, V, batch_size=None):
        nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())
        self.check_hierarchies(self._ion_classes, *nodes)
        for node in nodes:
            infos = tuple([self._get_imp(root).pack_info() for root in node.master_type.__args__])
            node.reset_state(V, *infos, batch_size)

    def check_hierarchy(self, roots, leaf):
        # 'master_type' should be a brainpy.mixin.JointType
        self._check_master_type(leaf)
        for cls in leaf.master_type.__args__:
            if not any([issubclass(root, cls) for root in roots]):
                raise TypeError(f'Type does not match. {leaf} requires a master with type '
                                f'of {leaf.master_type}, but the master type now is {roots}.')

    def add_elem(self, *elems, **elements):
        """Add new elements.

        Args:
          elements: children objects.
        """
        self.check_hierarchies(self._ion_classes, *elems, **elements)
        self.children.update(self.format_elements(IonChaDyn, *elems, **elements))
        for elem in tuple(elems) + tuple(elements.values()):
            for ion_root in elem.master_type.__args__:
                ion = self._get_imp(ion_root)
                ion.add_external_current(elem.name, self._get_ion_fun(ion, elem))

    def _get_ion_fun(self, ion, node):
        def fun(V, *args):
            infos = tuple([(ion.pack_info(*args)
                            if isinstance(ion, root) else
                            self._get_imp(root).pack_info())
                           for root in node.master_type.__args__])
            return node.current(V, *infos)

        return fun

    def _get_imp(self, cls):
        for ion in self.ions:
            if isinstance(ion, cls):
                return ion
        else:
            raise ValueError(f'No instance of {cls} is found.')

    def _check_master_type(self, leaf):
        if not isinstance(leaf.master_type, _JointGenericAlias):
            raise TypeError(f'{self.__class__.__name__} requires leaf nodes that have the master_type of '
                            f'"brainpy.mixin.JointType". However, we got {leaf.master_type}')


def mix_ions(*ions) -> MixIons:
    """Create mixed ions.

    Args:
      ions: Ion instances.

    Returns:
      Instance of MixIons.
    """
    for ion in ions:
        assert isinstance(ion, Ion), f'Must be instance of {Ion.__name__}. But got {type(ion)}'
    assert len(ions) > 0, ''
    return MixIons(*ions)


class Ion(IonChaDyn, Container, TreeNode):
    """The brainpy_object calcium dynamics.

    Args:
      size: The size of the simulation target.
      method: The numerical integration method.
      name: The name of the object.
      channels: The calcium dependent channels.
    """

    '''The type of the master object.'''
    master_type = HHTypedNeuron

    """Reversal potential."""
    E: Union[float, bm.Variable, bm.Array]

    """Calcium concentration."""
    C: Union[float, bm.Variable, bm.Array]

    def __init__(
        self,
        size: Shape,
        keep_size: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        **channels
    ):
        super().__init__(size, keep_size=keep_size, mode=mode, method=method, name=name)

        # Attribute of "Container"
        self.children = bm.node_dict(self.format_elements(IonChaDyn, **channels))
        self.external: Dict[str, Callable] = dict()  # not found by `.nodes()` or `.vars()`

    def update(self, V):
        for node in self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values():
            node.update(V, self.C, self.E)

    def current(self, V, C=None, E=None, external: bool = False):
        """Generate ion channel current.

        Args:
          V: The membrane potential.
          C: The given ion concentration.
          E: The given reversal potential.
          external: Include the external current.

        Returns:
          Current.
        """
        C = self.C if (C is None) else C
        E = self.E if (E is None) else E
        nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())
        self.check_hierarchies(type(self), *nodes)

        current = 0.
        if len(nodes) > 0:
            for node in nodes:
                current = current + node.current(V, C, E)
        if external:
            for key, node in self.external.items():
                current = current + node(V, C, E)
        return current

    def reset_state(self, V, batch_size=None):
        nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())
        self.check_hierarchies(type(self), *nodes)
        for node in nodes:
            node.reset_state(V, self.C, self.E, batch_size)

    def pack_info(self, C=None, E=None) -> Dict:
        if C is None:
            C = self.C
        if E is None:
            E = self.E
        return dict(C=C, E=E)

    def add_external_current(self, key: str, fun: Callable):
        if key in self.external:
            raise ValueError
        self.external[key] = fun
