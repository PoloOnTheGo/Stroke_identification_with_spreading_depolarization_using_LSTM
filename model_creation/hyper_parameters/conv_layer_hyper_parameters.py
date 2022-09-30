import json


class ConvLayerHyperparameter:
    def __init__(self, act_func=None, kernel_size=None, no_of_layers_and_neurons=None, dropout=None,
                 pooling=None, n_steps=None, n_length=None):
        self.act_func = act_func
        self.kernel_size = kernel_size
        self.no_of_layers_and_neurons = no_of_layers_and_neurons
        self.dropout = dropout
        self.pooling = pooling
        self.n_steps = n_steps
        self.n_length = n_length

    def str_format(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
