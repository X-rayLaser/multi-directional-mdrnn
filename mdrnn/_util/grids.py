import itertools

import numpy as np
import tensorflow as tf


class MultiDimensionalGrid(object):
    def __init__(self, grid_shape):
        self._shape = grid_shape
        self._grid = np.zeros(grid_shape).tolist()

    def get_positions(self):
        iterators = []
        for dim in self._shape:
            iterators.append(range(dim))

        for position in itertools.product(*iterators):
            yield position

    def set_grid(self, grid):
        self._grid = list(grid)

    def put_item(self, position, item):
        inner_list = self._get_inner_most_list(position)
        inner_list[position[-1]] = item

    def get_item(self, position):
        if len(position) != len(self._shape):
            raise InvalidPositionError()

        self._validate_position_dimensions(position)

        row = self._get_inner_most_list(position)
        return row[position[-1]]

    def get_sub_list(self, position):
        self._validate_position_dimensions(position)

        row = self._get_inner_most_list(position)
        return row[position[-1]]

    def _validate_position_dimensions(self, position):
        for i, xd in enumerate(position):
            if xd < 0 or xd >= self._shape[i]:
                raise PositionOutOfBoundsError()

    def _get_inner_most_list(self, position):
        if len(position) > 1:
            sub_list = self._grid[position[0]]
            for index in position[1:-1]:
                sub_list = sub_list[index]

            return sub_list
        else:
            return self._grid


class TensorGrid(MultiDimensionalGrid):
    def __init__(self, grid_shape, tensor_shape):
        super(TensorGrid, self).__init__(grid_shape)

        for position in self.get_positions():
            if tensor_shape[0] is None:
                tensor_shape = (1,) + tensor_shape[1:]
            zeros_tensor = tf.zeros(shape=tensor_shape)
            self.put_item(position, zeros_tensor)

        self._tensor_shape = tensor_shape

    @property
    def grid_shape(self):
        return self._shape

    def reduce_rank(self):
        if len(self.grid_shape) == 1:
            tensor = tf.stack(self._grid, axis=1)
            return NullGrid(tensor)

        new_tensor_shape = self._calculate_new_tensor_shape()
        new_grid_shape = self._calculate_new_grid_shape()

        new_grid = TensorGrid(grid_shape=new_grid_shape,
                              tensor_shape=new_tensor_shape)

        for position in new_grid.get_positions():
            tensor = self._fold_sub_list(position)
            new_grid.put_item(position, tensor)

        return new_grid

    def _calculate_new_tensor_shape(self):
        batch_size = self._tensor_shape[0]
        new_tensor_shape = (batch_size,) + (self.grid_shape[-2],) + self._tensor_shape[1:]
        return new_tensor_shape

    def _calculate_new_grid_shape(self):
        return self.grid_shape[:-1]

    def _fold_sub_list(self, position):
        inner_list = self.get_sub_list(position)
        return tf.stack(inner_list, axis=1)

    def to_tensor(self):
        new_grid = self.reduce_rank()

        while len(new_grid.grid_shape) > 0:
            new_grid = new_grid.reduce_rank()

        return new_grid.get_item((0,))


class NullGrid(MultiDimensionalGrid):
    def __init__(self, tensor):
        super(NullGrid, self).__init__(tuple())
        self._tensor = tensor

    @property
    def grid_shape(self):
        return tuple()

    def get_item(self, position):
        return self._tensor


class PositionOutOfBoundsError(Exception):
    pass


class InvalidPositionError(Exception):
    pass