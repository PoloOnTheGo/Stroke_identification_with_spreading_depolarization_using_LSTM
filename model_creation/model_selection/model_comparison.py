import numpy as np
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D

from data_preprocessing import data_constants as dc
from data_preprocessing.data_preprocessing_util import DataPreprocessingUtil
from data_preprocessing.data_processing_params import DataProcessingParams
from file_management import file_management_constant as fc
from file_management.file_management_util import FileManagementUtil
from model_creation import model_constants as mc
from model_creation.hyper_parameters.conv_layer_hyper_parameters import ConvLayerHyperparameter
from model_creation.hyper_parameters.dense_layer_hyper_parameters import DenseLayerHyperparameter
from model_creation.hyper_parameters.lstm_layer_hyper_parameters import LstmLayerHyperparameter
from model_creation.hyper_parameters.model_hyper_parameters import ModelHyperParameters
from model_creation.model_metrics_util import overall_model_analysis
from model_creation.sd_detection_model import SdDetectionModel
import random


def get_input_shape(dp, mp, trainX, testX):
    in_shape = ()
    if mp.model_type == mc.SIMPLE_LSTM:
        in_shape = (dp.time_window, dc.NO_OF_COLUMNS)
    elif mp.model_type == mc.CNN_LSTM:
        in_shape = (None, mp.clh.n_length, dc.NO_OF_COLUMNS)
        trainX = trainX.reshape((trainX.shape[0], mp.clh.n_steps, mp.clh.n_length, dc.NO_OF_COLUMNS))
        testX = testX.reshape((testX.shape[0], mp.clh.n_steps, mp.clh.n_length, dc.NO_OF_COLUMNS))
    elif mp.model_type == mc.CONV_LSTM:
        in_shape = (mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS)
        trainX = trainX.reshape((trainX.shape[0], mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS))
        testX = testX.reshape((testX.shape[0], mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS))
    return in_shape, trainX, testX


def add_simple_lstm_model_param(n_layers, dropout, model_list):
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=n_layers, dropout=dropout)
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh, dlh=dlh, verbose=1,
                                    batch_size=64, epochs=3)
    model_list.append(model_hp)


def add_cnn_lstm_model_param(n_layers, dropout, n_steps, n_length, model_list):
    clh = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_neurons=n_layers,
                                  dropout=dropout, pooling=MaxPooling1D(pool_size=2), n_steps=n_steps,
                                  n_length=n_length)
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=[100], dropout=Dropout(0.5))
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh, llh=llh, dlh=dlh, verbose=1,
                                    batch_size=64, epochs=3)
    model_list.append(model_hp)


def add_conv_lstm_model_param(n_layers, dropout, n_steps, n_length, model_list):
    clh = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_neurons=n_layers,
                                  dropout=dropout, pooling=MaxPooling1D(pool_size=2), n_steps=n_steps,
                                  n_length=n_length)
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=[100], dropout=Dropout(0.5))
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh, llh=llh, dlh=dlh,
                                    verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp)


def get_different_model_hyperparameter():
    model_list = []

    print("Model 1 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[100], dropout=Dropout(0.5), model_list=model_list)

    print("Model 2 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[128], dropout=Dropout(0.5), model_list=model_list)

    print("Model 3 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[256], dropout=Dropout(0.5), model_list=model_list)

    print("Model 4 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[256], dropout=Dropout(0.8), model_list=model_list)

    print("Model 5 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[64], dropout=Dropout(0.5), model_list=model_list)

    print("Model 6 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[64], dropout=Dropout(0.8), model_list=model_list)

    print("Model 7 (Simple LSTM) ===================================")
    add_simple_lstm_model_param(n_layers=[64], dropout=Dropout(0.25), model_list=model_list)

    print("Model 8 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=4, model_list=model_list)

    print("Model 9 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=8, model_list=model_list)

    print("Model 10 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=4, n_length=8, model_list=model_list)

    print("Model 11 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=16, model_list=model_list)

    print("Model 12 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=4, n_length=16, model_list=model_list)

    print("Model 13 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.25), n_steps=4, n_length=16, model_list=model_list)

    print("Model 14 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.8), n_steps=4, n_length=16, model_list=model_list)

    print("Model 15 (CNN LSTM)===================================")
    add_cnn_lstm_model_param(n_layers=[128, 128], dropout=Dropout(0.8), n_steps=2, n_length=16, model_list=model_list)

    print("Model 16 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=4, model_list=model_list)

    print("Model 17 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=8, model_list=model_list)

    print("Model 18 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=4, n_length=8, model_list=model_list)

    print("Model 19 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.25), n_steps=4, n_length=8, model_list=model_list)

    print("Model 20 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[128, 128], dropout=Dropout(0.25), n_steps=4, n_length=8, model_list=model_list)

    print("Model 21 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=16, model_list=model_list)

    print("Model 22 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=4, n_length=16, model_list=model_list)

    print("Model 23 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.25), n_steps=4, n_length=16, model_list=model_list)

    print("Model 24 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.8), n_steps=4, n_length=16, model_list=model_list)

    print("Model 25 (CONV LSTM)===================================")
    add_conv_lstm_model_param(n_layers=[128, 128], dropout=Dropout(0.8), n_steps=4, n_length=16, model_list=model_list)

    return model_list


def get_different_data_processing_params_set():
    params_set = []

    print("Data_processing_parameters 1===================================")
    params_1 = DataProcessingParams(before_sd_buffer=5, after_sd_buffer=5, time_window=3, time_window_shift=1)
    params_set.append(params_1)

    print("Data_processing_parameters 2===================================")
    params_2 = DataProcessingParams(before_sd_buffer=10, after_sd_buffer=10, time_window=8, time_window_shift=1)
    params_set.append(params_2)

    print("Data_processing_parameters 3===================================")
    params_3 = DataProcessingParams(before_sd_buffer=20, after_sd_buffer=20, time_window=16, time_window_shift=1)
    params_set.append(params_3)

    print("Data_processing_parameters 4===================================")
    params_4 = DataProcessingParams(before_sd_buffer=40, after_sd_buffer=40, time_window=32, time_window_shift=1)
    params_set.append(params_4)

    print("Data_processing_parameters 5===================================")
    params_5 = DataProcessingParams(before_sd_buffer=65, after_sd_buffer=65, time_window=32, time_window_shift=1)
    params_set.append(params_5)

    print("Data_processing_parameters 6===================================")
    params_6 = DataProcessingParams(before_sd_buffer=65, after_sd_buffer=65, time_window=64, time_window_shift=1)
    params_set.append(params_6)

    print("Data_processing_parameters 7===================================")
    params_7 = DataProcessingParams(before_sd_buffer=65, after_sd_buffer=65, time_window=64, time_window_shift=3)
    params_set.append(params_7)

    return params_set


def main():
    train_data_folder = fc.DATA_ROOT_PATH_FOR_MODEL_COMPARISON + fc.TRAIN_DATA_PATH
    test_data_folder = fc.DATA_ROOT_PATH_FOR_MODEL_COMPARISON + fc.TEST_DATA_PATH
    data_processing_params_set = get_different_data_processing_params_set()
    model_params_set = get_different_model_hyperparameter()
    for i in range(len(data_processing_params_set)):
        dp = data_processing_params_set[i]
        X_train, y_train = DataPreprocessingUtil(train_data_folder, dp).load_preprocessed_dataset()
        X_test, y_test = DataPreprocessingUtil(test_data_folder, dp).load_preprocessed_dataset()
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        for j in range(len(model_params_set)):
            mp: ModelHyperParameters = model_params_set[j]
            if mp.clh and (mp.clh.n_steps * mp.clh.n_length) != dp.time_window:
                continue
            print('============================= data_param_set: ' + str(i + 1) + ', model_param_set: '
                  + str(j + 1) + ' ===========================')
            input_shape, X_train_updated, X_test_updated = get_input_shape(dp, mp, X_train, X_test)
            sd_detection_model = SdDetectionModel(mp, input_shape)

            sd_detection_model.train_model(X_train_updated, y_train)

            fmu = FileManagementUtil()
            fmu.set_result_full_path(root=fc.DATA_ROOT_PATH_FOR_MODEL_COMPARISON, data_type=fc.TEST_DATA_PATH,
                                     dp_set_no=(i + 1), model_no=(j + 1))
            sd_detection_model.save_model(fmu)
            dp.save_data_params(fmu)

            accuracy = sd_detection_model.evaluate_model(X_test_updated, y_test)
            print(accuracy)

            prediction_file_name = fmu.get_result_full_file_name(patient_file_name='all_test_patient',
                                                                 file_type=fc.PREDICTION_CSV)
            conf_matrix_file_name = fmu.get_result_full_file_name(patient_file_name='all_test_patient',
                                                                  file_type=fc.CONF_MATRIX_JPG)
            metrics_file_name = fmu.get_result_full_file_name(patient_file_name='all_test_patient',
                                                              file_type=fc.METRICS_CSV)

            sd_detection_model.predict_sd(X_test, X_test_updated, y_test, prediction_file_name, conf_matrix_file_name)
            print('=====================================end of one model ===========================================')
            folder = r'/Users/poulamighosh/Documents/GitHub' \
                     r'/Stroke_identification_with_spreading_depolarization_using_LSTM/data/result/test_data '
            overall_model_analysis(folder, metrics_file_name)


if __name__ == "__main__":
    random.seed(10)
    main()
