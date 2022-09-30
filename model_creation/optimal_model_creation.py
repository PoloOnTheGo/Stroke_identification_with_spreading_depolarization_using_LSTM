import numpy as np
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D

from data_preprocessing import data_constants as dc
from data_preprocessing.data_preprocessing_util import DataPreprocessingUtil
from data_preprocessing.data_processing_params import DataProcessingParams
from file_management import file_management_constant as fc
from file_management.file_management_util import FileManagementUtil
from model_creation import model_constants as mc
from model_creation import model_util
from model_creation.hyper_parameters.conv_layer_hyper_parameters import ConvLayerHyperparameter
from model_creation.hyper_parameters.dense_layer_hyper_parameters import DenseLayerHyperparameter
from model_creation.hyper_parameters.lstm_layer_hyper_parameters import LstmLayerHyperparameter
from model_creation.hyper_parameters.model_hyper_parameters import ModelHyperParameters


def get_input_shape(dp, mp):
    in_shape = ()
    if mp.model_type == mc.SIMPLE_LSTM:
        in_shape = (dp.time_window, dc.NO_OF_COLUMNS)
    elif mp.model_type == mc.CNN_LSTM:
        in_shape = (None, mp.clh.n_length, dc.NO_OF_COLUMNS)
    elif mp.model_type == mc.CONV_LSTM:
        in_shape = (mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS)
    return in_shape


def get_reshaped_X(mp, X):
    if mp.model_type == mc.SIMPLE_LSTM:
        print("No reshape required")
    elif mp.model_type == mc.CNN_LSTM:
        X = X.reshape((X.shape[0], mp.clh.n_steps, mp.clh.n_length, dc.NO_OF_COLUMNS))
    elif mp.model_type == mc.CONV_LSTM:
        X = X.reshape((X.shape[0], mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS))
    return X


def get_simple_lstm_model_param(n_layers, dropout):
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=n_layers, dropout=dropout)
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh, dlh=dlh, verbose=1,
                                    batch_size=64, epochs=70)
    return model_hp


def get_cnn_lstm_model_param(n_layers, dropout, n_steps, n_length):
    clh = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_neurons=n_layers,
                                  dropout=dropout, pooling=MaxPooling1D(pool_size=2), n_steps=n_steps,
                                  n_length=n_length)
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=[100], dropout=Dropout(0.5))
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh, llh=llh, dlh=dlh, verbose=1,
                                    batch_size=64, epochs=70)
    return model_hp


def get_conv_lstm_model_param(n_layers, dropout, n_steps, n_length):
    clh = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_neurons=n_layers,
                                  dropout=dropout, pooling=MaxPooling1D(pool_size=2), n_steps=n_steps,
                                  n_length=n_length)
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=[100], dropout=Dropout(0.5))
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh, llh=llh, dlh=dlh,
                                    verbose=1, batch_size=64, epochs=70)
    return model_hp


def create_optimal_model():
    print("Data_processing_parameters 4===================================")
    data_params = DataProcessingParams(before_sd_buffer=40, after_sd_buffer=40, time_window=32, time_window_shift=1)

    print("Model 11 (CNN LSTM)===================================")
    model_params = get_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=16)

    input_shape = get_input_shape(data_params, model_params)

    sd_detection_model = model_util.create_model(model_params, input_shape)

    return sd_detection_model, model_params, data_params


def train_model(model, metrics, data_params):
    train_data_folder = fc.DATA_ROOT_PATH_TRAIN + fc.TRAIN_DATA_PATH
    X_train, y_train = DataPreprocessingUtil(train_data_folder, data_params).load_preprocessed_dataset()
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_train_updated = get_reshaped_X(metrics, X_train)
    model_util.train_model(model, metrics, X_train_updated, y_train)


def main():
    model, metrics, data_params = create_optimal_model()
    train_model(model, metrics, data_params)
    fmu = FileManagementUtil(fc.ROOT)
    model_util.save_model(model, fmu)
    data_params.save_data_params(fmu)


if __name__ == "__main__":
    main()
