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
import pytest

from brainpy.analysis import plotstyle


def test_set_markersize_updates_global():
    """Regression for P13-H1: ``set_markersize`` must update the module global
    ``_markersize`` (it previously assigned to a typo local ``__markersize``)."""
    original = plotstyle._markersize
    try:
        plotstyle.set_markersize(33)
        # per-key schema entries updated
        assert plotstyle.plot_schema[plotstyle.SADDLE_NODE]['markersize'] == 33
        # module global updated, not just the per-key dicts
        assert plotstyle._markersize == 33
    finally:
        plotstyle.set_markersize(original)


def test_set_markersize_type_check():
    with pytest.raises(TypeError):
        plotstyle.set_markersize(1.5)


def test_set_plot_schema_validations():
    with pytest.raises(TypeError):
        plotstyle.set_plot_schema(123)
    with pytest.raises(KeyError):
        plotstyle.set_plot_schema('not-a-real-fixed-point-type')
    # valid update
    plotstyle.set_plot_schema(plotstyle.SADDLE_NODE, color='black')
    assert plotstyle.plot_schema[plotstyle.SADDLE_NODE]['color'] == 'black'
