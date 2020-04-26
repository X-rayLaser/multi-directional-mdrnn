from unittest.case import TestCase
from mdrnn import MDRNN, InvalidParamsError


class MDRNNInitializationTests(TestCase):
    def make_rnn(self, **kwargs):
        return MDRNN(**kwargs)

    def assert_invalid_instances(self, *kwargs):
        for kwargs in kwargs:
            self.assertRaises(InvalidParamsError, lambda: self.make_rnn(**kwargs))

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


class MDGRUInitializationTests(TestCase):
    def make_rnn(self, **kwargs):
        from mdrnn._layers.gru import MDGRU
        return MDGRU(**kwargs)

    def assert_invalid_instances(self, *kwargs):
        for kwargs in kwargs:
            self.assertRaises(InvalidParamsError, lambda: self.make_rnn(**kwargs))

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

