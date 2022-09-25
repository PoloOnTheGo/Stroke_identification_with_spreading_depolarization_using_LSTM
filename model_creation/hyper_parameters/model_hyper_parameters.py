import json

from model_creation.hyper_parameters.conv_layer_hyper_parameters import ConvLayerHyperparameter
from model_creation.hyper_parameters.dense_layer_hyper_parameters import DenseLayerHyperparameter
from model_creation.hyper_parameters.lstm_layer_hyper_parameters import LstmLayerHyperparameter


class ModelHyperParameters:
    def __init__(self, model_type=None, threshold=None, llh: LstmLayerHyperparameter = None,
                 clh: ConvLayerHyperparameter = None, dlh: DenseLayerHyperparameter = None, verbose=None,
                 batch_size=None, epochs=None):
        self.model_type = model_type
        self.threshold = threshold

        self.llh: LstmLayerHyperparameter = llh
        self.clh: ConvLayerHyperparameter = clh
        self.dlh: DenseLayerHyperparameter = dlh

        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
