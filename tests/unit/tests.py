from unittest import TestCase
from mdrnn import MDRNN, InvalidParamsError, Direction


class DirectionValidationTestCase(TestCase):
    def setUp(self):
        self.input_dim = 3
        self.units = 3
        self.ndims = 1

    def create_mdrnn(self, direction):
        input_dimensions = [None] * self.ndims
        shape = tuple(input_dimensions + [self.input_dim])
        return MDRNN(units=self.units, input_shape=shape, direction=direction)

    def assert_invalid_direction(self, direction):
        self.assertRaises(
            InvalidParamsError,
            lambda: self.create_mdrnn(direction=direction)
        )

    def assert_valid_direction(self, direction):
        try:
            self.create_mdrnn(direction=direction)
        except InvalidParamsError:
            self.assertTrue(False, msg='Direction is not valid')


class DirectionValidationTests(DirectionValidationTestCase):
    def test_cannot_use_integer_for_direction(self):
        self.assert_invalid_direction(3)

    def test_cannot_use_list_for_direction(self):
        self.assert_invalid_direction([1, -1, -1])

    def test_length_of_direction_vector_must_match_dimensionality_of_input(self):
        self.assert_invalid_direction(Direction())
        self.assert_invalid_direction(Direction(1, 1))

    def test_with_valid_direction(self):
        self.assert_valid_direction(Direction(-1))

    def test_cannot_instantiate_direction_with_arbitrary_number(self):
        self.assertRaises(TypeError, lambda: Direction(8))
        self.assertRaises(TypeError, lambda: Direction(0))
        self.assertRaises(TypeError, lambda: Direction(-2, 1, -1))


class DirectionValidationInMultidimensionalRNN(DirectionValidationTestCase):
    def setUp(self):
        self.input_dim = 3
        self.units = 3
        self.ndims = 2

    def test_pass_invalid_direction_to_2drnn(self):
        self.assert_invalid_direction(Direction(1))

    def test_with_valid_direction(self):
        self.assert_valid_direction(Direction(-1, 1))


class OneDimensionalDirectionTests(TestCase):
    def test_left_to_right(self):
        direction = Direction(1)

        it = direction.iterate_over_positions(dim_lengths=[2])
        positions = list(it)

        expected = [(0,), (1,)]
        self.assertEqual(expected, positions)

    def test_right_to_left(self):
        direction = Direction(-1)

        it = direction.iterate_over_positions(dim_lengths=[3])
        positions = list(it)

        expected = [(2,), (1,), (0,)]
        self.assertEqual(expected, positions)


class OneDimensionalDirectionPreviousPositionTests(TestCase):
    def test_going_left_to_right(self):
        direction = Direction(1)
        position = (3,)
        self.assertEqual([(2,)], direction.get_previous_step_positions(position))

    def test_going_right_to_left(self):
        direction = Direction(-1)
        position = (3,)
        self.assertEqual([(4,)], direction.get_previous_step_positions(position))


class SouthEastPreviousPositionTests(TestCase):
    def setUp(self):
        self.position = (4, 2)

    def create_direction(self):
        return Direction(1, 1)

    def expected_result(self):
        return [(3, 2), (4, 1)]

    def test_previous_step_positions_returns_correct_result(self):
        direction = self.create_direction()
        expected_result = self.expected_result()
        actual = direction.get_previous_step_positions(self.position)
        self.assertEqual(expected_result, actual)


class SouthWestPreviousPositionTests(SouthEastPreviousPositionTests):
    def create_direction(self):
        return Direction(1, -1)

    def expected_result(self):
        return [(3, 2), (4, 3)]


class NorthEastPreviousPositionTests(SouthEastPreviousPositionTests):
    def create_direction(self):
        return Direction(-1, 1)

    def expected_result(self):
        return [(5, 2), (4, 1)]


class NorthWestPreviousPositionTests(SouthEastPreviousPositionTests):
    def create_direction(self):
        return Direction(-1, -1)

    def expected_result(self):
        return [(5, 2), (4, 3)]


class MultiDimensionalDirectionPreviousPositionTests(TestCase):
    def test(self):
        direction = Direction(1, -1, -1, 1)

        position = (3, 5, 2, 1)
        actual = direction.get_previous_step_positions(position)
        expected = [(2, 5, 2, 1), (3, 6, 2, 1), (3, 5, 3, 1), (3, 5, 2, 0)]
        self.assertEqual(expected, actual)


class NorthWestDirection(TestCase):
    def setUp(self):
        self.dim_lengths = [2, 3]
        self.direction = self.get_direction()
        self.expected = self.get_expected_result()

    def get_direction(self):
        return Direction(-1, -1)

    def get_expected_result(self):
        return [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]

    def test_direction(self):
        direction = self.direction

        it = direction.iterate_over_positions(dim_lengths=self.dim_lengths)
        positions = list(it)

        self.assertEqual(self.expected, positions)


class NorthEastDirection(NorthWestDirection):
    def get_direction(self):
        return Direction(-1, 1)

    def get_expected_result(self):
        return [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)]


class SouthWestDirection(NorthWestDirection):
    def get_direction(self):
        return Direction(1, -1)

    def get_expected_result(self):
        return [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)]


class SouthEastDirection(NorthWestDirection):
    def get_direction(self):
        return Direction(1, 1)

    def get_expected_result(self):
        return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


class DirectionsIterationTests(TestCase):
    def test_getting_2_directions_for_1_dimensional_space(self):
        directions = Direction.get_all_directions(ndims=1)
        self.assertEqual([Direction(1), Direction(-1)], directions)

    def test_getting_4_directions_for_2_dimensional_space(self):
        directions = Direction.get_all_directions(ndims=2)
        expected = [Direction(1, 1), Direction(1, -1), Direction(-1, 1), Direction(-1, -1)]
        self.assertEqual(expected, directions)

    def test_getting_8_directions_for_3_dimensional_space(self):
        directions = Direction.get_all_directions(ndims=3)
        expected = [Direction(1, 1, 1), Direction(1, 1, -1),
                    Direction(1, -1, 1), Direction(1, -1, -1),
                    Direction(-1, 1, 1), Direction(-1, 1, -1),
                    Direction(-1, -1, 1), Direction(-1, -1, -1)]
        self.assertEqual(expected, directions)

    def test_getting_2_to_the_n_directions_for_n_dimensional_space(self):
        n = 5
        directions = Direction.get_all_directions(ndims=n)
        self.assertEqual(2**n, len(directions))

    def test_no_duplicate_directions(self):
        n = 7
        directions = Direction.get_all_directions(ndims=n)
        unique = set(directions)
        self.assertEqual(len(directions), len(unique))


# todo 1 dimensional rnn, specify iteration direction
# todo bidirectional 1 dimensional rnn
# todo 2 dimensional rnn
# todo 2 dimensional rnn 4 directional rnn
# todo n dimensional rnn
# todo n dimensional rnn multidirectional
