import json


class DenseLayerHyperparameter:
    def __init__(self, no_of_neurons, act_func):
        self.act_func = act_func
        self.no_of_neurons = no_of_neurons

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
