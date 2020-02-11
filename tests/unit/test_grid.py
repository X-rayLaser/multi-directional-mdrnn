from unittest import TestCase
from mdrnn import MultiDimensionalGrid, PositionOutOfBoundsError, InvalidPositionError
import tensorflow as tf
import numpy as np


class TensorGridTests(TestCase):
    def setUp(self):
        self.grid = MultiDimensionalGrid(grid_shape=(4, 2))

    def test_get_tensor_after_initialization_returns_zeros(self):
        self.grid = MultiDimensionalGrid(grid_shape=(2,))

        self.assertEqual(0, self.grid.get_item((0,)))

    def test_get_item_using_negative_integers(self):
        self.assertRaises(
            PositionOutOfBoundsError, lambda: self.grid.get_item((2, -1))
        )

    def test_get_item_using_out_of_bound_position(self):
        self.assertRaises(
            PositionOutOfBoundsError, lambda: self.grid.get_item((2, 12))
        )

        self.assertRaises(
            PositionOutOfBoundsError, lambda: self.grid.get_item((4, 0))
        )

    def test_get_item_when_passing_less_than_required_position_components(self):
        self.assertRaises(
            InvalidPositionError, lambda: self.grid.get_item((2, ))
        )

    def test_get_item_when_passing_more_than_required_position_components(self):
        self.assertRaises(
            InvalidPositionError, lambda: self.grid.get_item((2, 1, 0, 0))
        )


class RetrievalTests(TestCase):
    def test_1d(self):
        grid = MultiDimensionalGrid(grid_shape=(3, ))
        grid.set_grid([1, 2, 3])

        self.assertEqual(2, grid.get_item((1, )))

    def test_retrieve_from_2d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 3))

        a = [[1, 2, 3],
             [4, 5, 6]]
        grid.set_grid(a)

        self.assertEqual(4, grid.get_item((1, 0)))
        self.assertEqual(3, grid.get_item((0, 2)))

    def test_retrieve_from_3d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 2, 3))

        mtx1 = [[1, 2, 3],
                [4, 5, 6]]
        mtx2 = [[7, 8, 9],
                [10, 11, 12]]

        a = [mtx1, mtx2]
        grid.set_grid(a)

        self.assertEqual(7, grid.get_item((1, 0, 0)))
        self.assertEqual(12, grid.get_item((1, 1, 2)))

    def test_retrieve_from_6d_grid(self):
        shape = (2, 3, 4, 5, 6, 7)
        grid = MultiDimensionalGrid(grid_shape=shape)

        a = np.arange(2 * 3 * 4 * 5 * 6 * 7).reshape(shape).tolist()

        grid.set_grid(a)

        self.assertEqual(a[1][2][3][4][5][6], grid.get_item((1, 2, 3, 4, 5, 6)))


class StoringTests(TestCase):
    def setUp(self):
        self.grid = MultiDimensionalGrid(grid_shape=(4, 2))

    def test_store_and_retrieve_item(self):
        position = (2, 1)

        t = 2

        self.grid.put_item(position, t)

        actual = self.grid.get_item(position)
        self.assertEqual(t, actual)

    def test_store_and_retrieve_two_items(self):
        pos1 = (2, 1)
        pos2 = (1, 0)

        t1 = 4
        t2 = 7

        self.grid.put_item(pos1, t1)
        self.grid.put_item(pos2, t2)

        actual = self.grid.get_item(pos1)
        self.assertEqual(t1, actual)

        actual = self.grid.get_item(pos2)
        self.assertEqual(t2, actual)
