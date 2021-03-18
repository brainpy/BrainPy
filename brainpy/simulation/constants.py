# -*- coding: utf-8 -*-


NEU_GROUP_TYPE = 'NeuGroup'  # name of the neuron group
SYN_CONN_TYPE = 'SynConn'  # name of the synapse connection
TWO_END_TYPE = 'TwoEndConn'  # name of the two-end synaptic connection
SUPPORTED_TYPES = [NEU_GROUP_TYPE, SYN_CONN_TYPE, TWO_END_TYPE]

# input operations
SUPPORTED_INPUT_OPS = {'-': 'sub',
                       '+': 'add',
                       'x': 'mul',
                       '*': 'mul',
                       '/': 'div',
                       '=': 'assign'}
