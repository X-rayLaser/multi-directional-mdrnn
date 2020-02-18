import itertools


class Direction:
    @staticmethod
    def south_east():
        return Direction(1, 1)

    @staticmethod
    def south_west():
        return Direction(1, -1)

    @staticmethod
    def north_east():
        return Direction(-1, 1)

    @staticmethod
    def north_west():
        return Direction(-1, -1)

    @staticmethod
    def get_all_directions(ndims):
        direction_options = [1, -1]
        iterables = [direction_options] * ndims
        iterators = itertools.product(*iterables)

        directions = []

        for option in iterators:
            directions.append(Direction(*option))
        return directions

    def __init__(self, *directions):
        self._dirs = list(directions)
        for d in self._dirs:
            if d not in [-1, 1]:
                raise TypeError('data type not understood')

    @property
    def dimensions(self):
        return len(self._dirs)

    def iterate_over_positions(self, dim_lengths):
        axes = []

        for i, dim_len in enumerate(dim_lengths):
            step_size = self._dirs[i]
            if step_size > 0:
                index_iter = range(dim_len)
            else:
                index_iter = range(dim_len - 1, -1, -1)

            axes.append(index_iter)

        return list(itertools.product(*axes))

    def get_previous_step_positions(self, current_position):
        res = []
        for d in range(len(self._dirs)):
            position = self._get_previous_position_along_axis(current_position, d)
            res.append(position)

        return res

    def get_final_position(self):
        position = []
        for d in self._dirs:
            if d == 1:
                position.append(-1)
            elif d == -1:
                position.append(0)
            else:
                raise Exception('Should not get here')
        return tuple(position)

    def _get_previous_position_along_axis(self, position, axis):
        step = self._dirs[axis]
        return position[:axis] + (position[axis] - step,) + position[axis + 1:]

    def __eq__(self, other):
        return self._dirs == other._dirs

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        params_string = ','.join(map(str, self._dirs))
        return 'Direction ({})'.format(params_string)