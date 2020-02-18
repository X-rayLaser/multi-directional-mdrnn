from unittest import TestCase
from mdrnn import MultiDimensionalGrid, TensorGrid
from mdrnn import PositionOutOfBoundsError, InvalidPositionError
import tensorflow as tf
import numpy as np


class ExceptionsTests(TestCase):
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


class GridPositionsTests(TestCase):
    def test_for_1d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(3, ))
        all_positions = list(grid.get_positions())
        self.assertEqual([(0,), (1,), (2,)], all_positions)

    def test_for_2d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 3))
        all_positions = list(grid.get_positions())
        expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        self.assertEqual(expected, all_positions)


def get_inner_most_list(a, position):
    if len(position) == 1:
        return a

    sublist = a[position[0]]
    sub_position = position[1:]
    return get_inner_most_list(sublist, sub_position)


class RetrievalTests(TestCase):
    def assert_all_retrievals_are_correct(self, grid, expected_list, shape):
        for position in grid.get_positions():
            actual_item = grid.get_item(position)
            inner_list = get_inner_most_list(expected_list, position)
            expected_item = inner_list[position[-1]]
            self.assertEqual(expected_item, actual_item)

    def test_1d(self):
        grid = MultiDimensionalGrid(grid_shape=(3, ))
        grid.set_grid([1, 2, 3])

        self.assertEqual(2, grid.get_item((1, )))

    def test_all_retrievals_from_1d_grid_return_correct_items(self):
        grid = MultiDimensionalGrid(grid_shape=(3, ))

        a = [1, 2, 3]
        grid.set_grid(a)

        for i in range(3):
            self.assertEqual(a[i], grid.get_item((i,)))

    def test_retrieve_from_2d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 3))

        a = [[1, 2, 3],
             [4, 5, 6]]
        grid.set_grid(a)

        self.assertEqual(4, grid.get_item((1, 0)))
        self.assertEqual(3, grid.get_item((0, 2)))

    def test_all_retrievals_from_2d_grid_return_correct_items(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 3))

        a = [[1, 2, 3],
             [4, 5, 6]]
        grid.set_grid(a)

        for i in range(2):
            for j in range(3):
                self.assertEqual(a[i][j], grid.get_item((i, j)))

    def test_all_retrievals_from_3d_grid_return_correct_items(self):
        shape = (5, 3, 2)
        grid = MultiDimensionalGrid(grid_shape=shape)

        a = np.arange(5*3*2).reshape(shape).tolist()
        grid.set_grid(a)

        self.assert_all_retrievals_are_correct(grid, a, shape)

    def test_retrieve_from_10d_grid(self):
        shape = tuple([2] * 10)
        grid = MultiDimensionalGrid(grid_shape=shape)
        a = np.arange(2**10).reshape(shape).tolist()
        grid.set_grid(a)

        self.assert_all_retrievals_are_correct(grid, a, shape)

    def test_store_and_retrieve_in_3d_grid_after_initialization(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 4, 3))
        pos = (1, 2, 2)

        self.assertEqual(0, grid.get_item(pos))


class StoringTests(TestCase):
    def assert_retrieved_items_match_stored_ones(self, grid, a, shape):
        for position in grid.get_positions():
            inner_list = get_inner_most_list(a, position)
            item = inner_list[position[-1]]
            grid.put_item(position, item)

        for position in grid.get_positions():
            inner_list = get_inner_most_list(a, position)
            expected_item = inner_list[position[-1]]
            item = grid.get_item(position)
            self.assertEqual(expected_item, item)

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

    def test_filling_2d_grid_completely_and_retrieving_all_entries(self):
        a = np.arange(4*2).reshape(4, 2)
        self.assert_retrieved_items_match_stored_ones(self.grid, a, (4, 2))

    def test_store_and_retrieve_in_3d_grid(self):
        grid = MultiDimensionalGrid(grid_shape=(2, 4, 3))
        pos = (1, 2, 2)

        item = 35
        self.assertEqual(0, grid.get_item(pos))
        grid.put_item(pos, item)
        self.assertEqual(item, grid.get_item(pos))

    def test_filling_10d_grid_and_retrieving_all_entries_back(self):
        shape = tuple([2]*10)
        grid = MultiDimensionalGrid(shape)
        a = np.arange(2**10).reshape(shape).tolist()

        self.assert_retrieved_items_match_stored_ones(grid, a, shape)


class TensorGridTests(TestCase):
    def test_contains_zero_tensors_of_correct_shape_initially(self):
        tensor_shape = (2, 3, 2, 4)
        grid = TensorGrid(grid_shape=(1, 2), tensor_shape=tensor_shape)

        tensor = grid.get_item((0, 0))
        np.testing.assert_almost_equal(np.zeros(tensor_shape), tensor.numpy())

        tensor = grid.get_item((0, 1))
        np.testing.assert_almost_equal(np.zeros(tensor_shape), tensor.numpy())

    def test_store_and_retrive_tensor(self):
        grid = TensorGrid((3, 5, 4), tensor_shape=(2, ))

        t = tf.constant([2, 3])

        position = (0, 2, 2)
        grid.put_item(position, t)

        np.testing.assert_almost_equal(t.numpy(), grid.get_item(position).numpy(), 6)

    def test_reduce_1d_grid_to_single_tensor_of_right_shape(self):
        grid = TensorGrid((3, ), tensor_shape=(2, 5))
        null_grid = grid.reduce_rank()
        self.assertTupleEqual(tuple(), null_grid.grid_shape)
        self.assertEqual((2, 3, 5), null_grid.get_item(0).shape)

        grid = TensorGrid((3, ), tensor_shape=(2,))
        null_grid = grid.reduce_rank()
        self.assertEqual((2, 3), null_grid.get_item(0).shape)

    def test_reduce_1d_grid_produces_correct_tensor(self):
        grid = TensorGrid((2, ), tensor_shape=(3,))
        a1 = np.arange(3)
        a2 = np.arange(3) + 3

        grid.put_item((0,), tf.constant(a1))
        grid.put_item((1,), tf.constant(a2))

        null_grid = grid.reduce_rank()
        actual = null_grid.get_item(0)

        expected = np.array([
            [0, 3],
            [1, 4],
            [2, 5]
        ])

        np.testing.assert_almost_equal(expected, actual.numpy())

    def test_reduce_2d_grid_to_1d_grid(self):
        grid_2d = TensorGrid((2, 3), tensor_shape=(4, 5))
        grid_1d = grid_2d.reduce_rank()
        self.assertTupleEqual((2,), grid_1d.grid_shape)
        self.assertEqual((4, 3, 5), grid_1d.get_item((0,)).shape)
        self.assertEqual((4, 3, 5), grid_1d.get_item((1,)).shape)

    def test_reduce_3d_grid_to_2d_grid_of_tensors(self):
        grid_3d = TensorGrid((2, 3, 4), tensor_shape=(10, 6))
        grid_2d = grid_3d.reduce_rank()
        self.assertTupleEqual((2, 3), grid_2d.grid_shape)

        for i in range(2):
            for j in range(3):
                self.assertEqual((10, 4, 6), grid_2d.get_item((i, j)).shape)

    def test_reduce_3d_grid_to_1d_grid_of_tensors(self):
        grid_3d = TensorGrid((2, 3, 4), tensor_shape=(10, 6))
        null_grid = grid_3d.reduce_rank().reduce_rank().reduce_rank()
        self.assertTupleEqual((), null_grid.grid_shape)

        self.assertEqual((10, 2, 3, 4, 6), null_grid.get_item(0).shape)


class TensorGridToTensorTests(TestCase):
    def test_convert_1d_grid(self):
        tensor_shape = (2, 4)

        grid = TensorGrid(grid_shape=(3,), tensor_shape=tensor_shape)

        self.assertEqual((2, 3, 4), grid.to_tensor().shape)

    def test_convert_2d_grid(self):
        tensor_shape = (2, 4)

        grid = TensorGrid(grid_shape=(3, 5), tensor_shape=tensor_shape)

        self.assertEqual((2, 3, 5, 4), grid.to_tensor().shape)

    def test_convert_3d_grid(self):
        tensor_shape = (2, 4)

        grid = TensorGrid(grid_shape=(3, 5, 6), tensor_shape=tensor_shape)

        self.assertEqual((2, 3, 5, 6, 4), grid.to_tensor().shape)
