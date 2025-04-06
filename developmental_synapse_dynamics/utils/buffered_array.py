# -*- coding: utf-8 -*-
"""
BufferedArray
=============

A simple class for a numpy array that extends size dynamically without reallocating memory frequently.

Note
----
For simplicity extension is handeled along the first axis only.

Examples
--------
>>> from utils.buffered_array import BufferedArray
>>> a = BufferedArray(shape=(1,4), dtype=int)
>>> a
array([], shape=(0, 4), dtype=int64)

>>> a.capacity
1

>>> a.append([1,2,3,4])
>>> a
array([[1, 2, 3, 4]])

>>> a.append([5,6,7,8])
>>> a[1]  # noqa
array([5, 6, 7, 8])

>>> a[:]
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

>>> a.capacity
2

>>> a.append([5,6,7,8])
>>> a.capacity
4

>>> len(a)
3

>>> a.shape
(3, 4)

>>> import numpy as np
>>> a.extend(np.ones((50,4)))
>>> a.shape
(53, 4)

>>> a.capacity
64

>>> a.delete([3,4,5])
>>> a.shape
(50,4)
"""
__project__ = 'Developmental Synapse Remodeling'
__author__ = 'Christoph Kirst <christoph.kirst@ucsf.edu>'
__copyright__ = 'Copyright © 2025 by Christoph Kirst'


import numpy as np


class BufferedArray:
    """Simple numpy buffer"""
    def __init__(self, shape, dtype=float, fill_value=0):
        self.data: np.ndarray = np.full(shape, dtype=dtype, fill_value=fill_value)
        self.size: int = 0
        self.capacity: int = len(self.data)
        self.fill_value = fill_value

    @property
    def shape(self):
        return (self.size,) + self.data.shape[1:]

    def __len__(self):
        return self.size

    def _index(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        index0 = index[0]
        if isinstance(index0, np.ndarray):
            if not index0.ndim == 1 or not np.all(np.logical_and(-self.size <= index0, index0 < self.size)):
                raise IndexError(f"index {index0} along axis 0 out of bounds")
        elif isinstance(index0, slice) and index0 == slice(None):
            index0 = slice(None, self.size, None)
        elif isinstance(index0, slice) or (isinstance(index0, int) and index0 < 0):
            index0 = range(self.size)[index0]
        if isinstance(index0, (int, tuple, list, np.ndarray)):
            check = np.array(index0, ndmin=1)
            if not np.all(np.logical_and(0 <= check, check < self.size)):
                raise IndexError(f"index {index0} along axis 0 out of bounds")

        return (index0,) + index[1:]

    def __getitem__(self, index):
        return self.data[self._index(index)]

    def __setitem__(self, index, value):
        self.data[self._index(index)] = value

    def append(self, value):
        if self.size == self.capacity:
            self.resize(2 * self.capacity)
        self.data[self.size] = value
        self.size += 1

    def extend(self, value: np.ndarray):
        if self.shape[1:] != value.shape[1:]:
            raise ValueError(f"cannot concatenate arrays of shape {self.shape} and {value.shape} along axis=0")
        size = self.size
        capacity = self.capacity
        new_size = size + len(value)
        if new_size > capacity:
            while new_size > capacity:
                capacity *= 2
            self.resize(capacity)
        self.data[size: new_size] = value
        self.size = new_size

    def delete(self, indices):
        n = len(indices)
        if n == 0:
            return
        self.data[:-n] = np.delete(self.data, indices, axis=0)
        self.size -= n

    def resize(self, capacity: int):
        add_capacity = capacity - self.capacity
        if add_capacity < 0:
            raise RuntimeError("can only increase capacity.")
        shape = self.shape
        new_shape = (add_capacity,) + shape[1:]
        self.data = np.concatenate(
            [self.data, np.full(new_shape, dtype=self.data.dtype, fill_value=self.fill_value)], axis=0)
        self.capacity += add_capacity

    def max(self):
        return np.max(self.data[:])

    def __repr__(self):
        return np.ndarray.__repr__(self.data[:self.size])
