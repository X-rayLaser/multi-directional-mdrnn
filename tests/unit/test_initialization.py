from unittest.case import TestCase

from mdrnn import InvalidParamsError, MDRNN


class MDRNNInitializationTests(TestCase):
    def assert_invalid_instances(self, *kwargs):
        for kwargs in kwargs:
            self.assertRaises(InvalidParamsError, lambda: MDRNN(**kwargs))

    def test_with_invalid_input_dim(self):
        self.assert_invalid_instances(dict(units=10, input_shape=(3, 5, -1)),
                                      dict(units=10, input_shape=(3, 5, 0)),
                                      dict(units=10, input_shape=(3, 5, 10**10)))

    def test_with_invalid_units(self):
        self.assert_invalid_instances(dict(units=-1, input_shape=(3, 5, 1)),
                                      dict(units=0, input_shape=(3, 5, 1)),
                                      dict(units=10**10, input_shape=(3, 5, 1)))

    def test_with_invalid_number_of_dimensions(self):
        args = tuple([1] * 10**4)
        self.assert_invalid_instances(dict(units=1, input_shape=(1,)),
                                      dict(units=1, input_shape=args))