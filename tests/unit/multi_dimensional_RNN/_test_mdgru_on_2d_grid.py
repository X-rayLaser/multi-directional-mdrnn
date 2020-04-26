from .test_mdrnn_on_2d_grid import Degenerate2DInputToMDRNNTests, \
    OutputShapeGiven2DTests, OutputShapeGiven6DInputTests
import tensorflow as tf
from mdrnn import MDGRU


class Degenerate2DInputToMDGRUTests(Degenerate2DInputToMDRNNTests):
    def create_mdrnn(self, **kwargs):
        return MDGRU(**kwargs)

    def create_keras_rnn(self, **kwargs):
        return tf.keras.layers.GRU(implementation=1, reset_after=False, **kwargs)


class MDGRUOutputShapeGiven2DTests(OutputShapeGiven2DTests):
    def get_rnn_class(self):
        return MDGRU


class MDGRUOutputShapeGiven6DInputTests(OutputShapeGiven6DInputTests):
    def get_rnn_class(self):
        return MDGRU
