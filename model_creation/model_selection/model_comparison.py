from keras.layers import Dropout
import numpy as np

from data_preprocessing import data_constants as dc
from data_preprocessing.data_preprocessing_util import DataPreprocessingUtil
from data_preprocessing.data_processing_params import DataProcessingParams
from model_creation import model_constants as mc
from model_creation.hyper_parameters.dense_layer_hyper_parameters import DenseLayerHyperparameter
from model_creation.hyper_parameters.lstm_layer_hyper_parameters import LstmLayerHyperparameter
from model_creation.hyper_parameters.model_hyper_parameters import ModelHyperParameters
from model_creation.sd_detection_model import SdDetectionModel


def get_different_model_hyperparameter():
    model_list = []

    print("Model 1===================================")
    llh_1 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_1 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_1 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_1, dlh=dlh_1, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_1)

    return model_list


def get_different_data_processing_params_set():
    params_set = []

    print("Data_processing_parameters 1===================================")
    params_1 = DataProcessingParams(before_sd_buffer=5, after_sd_buffer=5, time_window=3, time_window_shift=1)
    params_set.append(params_1)

    return params_set


def get_input_shape(dp, mp):
    in_shape = ()
    if mp.model_type == mc.SIMPLE_LSTM:
        in_shape = (dp.time_window, dc.NO_OF_COLUMNS)
    elif mp.model_type == mc.CNN_LSTM:
        in_shape = (None, mp.clh.n_length, dc.NO_OF_COLUMNS)
    elif mp.model_type == mc.CONV_LSTM:
        in_shape = (mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS)
    return in_shape


if __name__ == "__main__":
    train_data_folder = '../../data/train_data'
    test_data_folder = '../../data/test_data'

    data_processing_params_set = get_different_data_processing_params_set()
    model_params_set = get_different_model_hyperparameter()

    for i in range(len(data_processing_params_set)):
        data_params = data_processing_params_set[i]
        for j in range(len(model_params_set)):
            X_train, y_train = DataPreprocessingUtil(train_data_folder, data_params).load_preprocessed_dataset()
            input_shape = get_input_shape(data_params, model_params_set[j])
            sd_detection_model = SdDetectionModel(model_params_set[j], input_shape)
            sd_detection_model.save_model()

            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)

            sd_detection_model.train_model(X_train, y_train)

            X_test, y_test = DataPreprocessingUtil(test_data_folder, data_params).load_preprocessed_dataset()
            accuracy = sd_detection_model.evaluate_model(X_test, y_test)
            print(accuracy)
            sd_detection_model.predict_sd(X_test, y_test, (j + 1), "all_test_patient", "test", (i+1))
