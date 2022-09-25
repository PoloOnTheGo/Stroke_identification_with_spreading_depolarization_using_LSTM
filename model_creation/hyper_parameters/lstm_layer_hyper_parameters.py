import json


class LstmLayerHyperparameter:
    def __init__(self, no_of_layers_and_filters=None, dropout=None):
        self.no_of_layers_and_filters = no_of_layers_and_filters
        self.dropout = dropout

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
