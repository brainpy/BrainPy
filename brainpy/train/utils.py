# -*- coding: utf-8 -*-


msg = '''

Given the data with (x_train, y_train) is no longer supported.

Please control your data by yourself. For example, using `torchvision` or `tensorflow-datasets`. 

A simple way to convert your `(x_train, y_train)` data is defining it as a python function:

.. code::

   def data(batch_size):
     x_data = bm.random.shuffle(x_data, key=123)
     y_data = bm.random.shuffle(y_data, key=123)
     for i in range(0, x_data.shape[0], batch_size):
       yield x_data[i: i + batch_size], y_data[i: i + batch_size], 

'''
