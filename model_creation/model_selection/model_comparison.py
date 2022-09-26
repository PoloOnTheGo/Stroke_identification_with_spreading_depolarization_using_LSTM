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
from model_creation.sd_detection_model import SdDetectionModel


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


def get_different_model_hyperparameter():
    model_list = []

    print("Model 1 (Simple LSTM) ===================================")
    llh_1 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_1 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_1 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_1, dlh=dlh_1, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_1)

    print("Model 2 (Simple LSTM) ===================================")
    llh_2 = LstmLayerHyperparameter(no_of_layers_and_filters=[128], dropout=Dropout(0.5))
    dlh_2 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_2 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_2, dlh=dlh_2, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_2)

    print("Model 3 (Simple LSTM) ===================================")
    llh_3 = LstmLayerHyperparameter(no_of_layers_and_filters=[256], dropout=Dropout(0.5))
    dlh_3 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_3 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_3, dlh=dlh_3, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_3)

    print("Model 4 (Simple LSTM) ===================================")
    llh_4 = LstmLayerHyperparameter(no_of_layers_and_filters=[256], dropout=Dropout(0.8))
    dlh_4 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_4 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_4, dlh=dlh_4, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_4)

    print("Model 5 (Simple LSTM) ===================================")
    llh_5 = LstmLayerHyperparameter(no_of_layers_and_filters=[64], dropout=Dropout(0.5))
    dlh_5 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_5 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_5, dlh=dlh_5, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_5)

    print("Model 6 (Simple LSTM) ===================================")
    llh_6 = LstmLayerHyperparameter(no_of_layers_and_filters=[64], dropout=Dropout(0.8))
    dlh_6 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_6 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_6, dlh=dlh_6, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_6)

    print("Model 7 (Simple LSTM) ===================================")
    llh_7 = LstmLayerHyperparameter(no_of_layers_and_filters=[64], dropout=Dropout(0.25))
    dlh_7 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_7 = ModelHyperParameters(model_type=mc.SIMPLE_LSTM, threshold=0.5, llh=llh_7, dlh=dlh_7, verbose=1,
                                      batch_size=64, epochs=3)
    model_list.append(model_hp_7)

    print("Model 8 (CNN LSTM)===================================")
    clh_8 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[64, 64],
                                    dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=2, n_length=4)
    llh_8 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_8 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_8 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_8, llh=llh_8, dlh=dlh_8,
                                      verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_8)

    print("Model 9 (CNN LSTM)===================================")
    clh_9 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[64, 64],
                                    dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=2, n_length=8)
    llh_9 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_9 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_9 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_9, llh=llh_9, dlh=dlh_9,
                                      verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_9)

    print("Model 10 (CNN LSTM)===================================")
    clh_10 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_10 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_10 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_10 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_10, llh=llh_10, dlh=dlh_10,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_10)

    print("Model 11 (CNN LSTM)===================================")
    clh_11 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.25), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_11 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_11 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_11 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_11, llh=llh_11, dlh=dlh_11,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_11)

    print("Model 12 (CNN LSTM)===================================")
    clh_12 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.8), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_12 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_12 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_12 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_12, llh=llh_12, dlh=dlh_12,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_12)

    print("Model 13 (CNN LSTM)===================================")
    clh_13 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_filters=[128, 128],
                                     dropout=Dropout(0.8), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_13 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_13 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_13 = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh_13, llh=llh_13, dlh=dlh_13,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_13)

    print("Model 14 (CONV LSTM)===================================")
    clh_14 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=2, n_length=4)
    llh_14 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_14 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_14 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_14, llh=llh_14, dlh=dlh_14,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_14)

    print("Model 15 (CNN LSTM)===================================")
    clh_15 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=2, n_length=8)
    llh_15 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_15 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_15 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_15, llh=llh_15, dlh=dlh_15,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_15)

    print("Model 16 (CNN LSTM)===================================")
    clh_16 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.5), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_16 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_16 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_16 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_16, llh=llh_16, dlh=dlh_16,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_16)

    print("Model 17 (CNN LSTM)===================================")
    clh_17 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.25), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_17 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_17 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_17 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_17, llh=llh_17, dlh=dlh_17,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_17)

    print("Model 18 (CNN LSTM)===================================")
    clh_18 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2), no_of_layers_and_filters=[64, 64],
                                     dropout=Dropout(0.8), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_18 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_18 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_18 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_18, llh=llh_18, dlh=dlh_18,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_18)

    print("Model 19 (CNN LSTM)===================================")
    clh_19 = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=(1, 2),
                                     no_of_layers_and_filters=[128, 128],
                                     dropout=Dropout(0.8), pooling=MaxPooling1D(pool_size=2), n_steps=4, n_length=16)
    llh_19 = LstmLayerHyperparameter(no_of_layers_and_filters=[100], dropout=Dropout(0.5))
    dlh_19 = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp_19 = ModelHyperParameters(model_type=mc.CONV_LSTM, threshold=0.5, clh=clh_19, llh=llh_19, dlh=dlh_19,
                                       verbose=1, batch_size=64, epochs=3)
    model_list.append(model_hp_19)

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
    params_6 = DataProcessingParams(before_sd_buffer=65, after_sd_buffer=65, time_window=32, time_window_shift=3)
    params_set.append(params_6)

    return params_set


def main():
    train_data_folder = fc.DATA_ROOT_PATH + fc.TRAIN_DATA_PATH
    test_data_folder = fc.DATA_ROOT_PATH + fc.TEST_DATA_PATH
    data_processing_params_set = get_different_data_processing_params_set()
    model_params_set = get_different_model_hyperparameter()
    for i in range(len(data_processing_params_set)):
        dp = data_processing_params_set[i]
        X_train, y_train = DataPreprocessingUtil(train_data_folder, dp).load_preprocessed_dataset()
        X_test, y_test = DataPreprocessingUtil(test_data_folder, dp).load_preprocessed_dataset()
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        for j in range(len(model_params_set)):
            mp = model_params_set[j]

            if mp.clh:
                print(i, j, mp.clh.n_steps, mp.clh.n_length, dp.time_window)
                if (mp.clh.n_steps * mp.clh.n_length) != dp.time_window:
                    print('skip')
                    continue
            input_shape, X_train_updated, X_test_updated = get_input_shape(dp, mp, X_train, X_test)
            print(input_shape)
            sd_detection_model = SdDetectionModel(mp, input_shape)

            fmu = FileManagementUtil()
            fmu.set_result_full_path(data_type=fc.TEST_DATA_PATH, dp_set_no=(i + 1), model_no=(j + 1))
            sd_detection_model.save_model(fmu)
            dp.save_data_params(fmu)

            sd_detection_model.train_model(X_train_updated, y_train)

            accuracy = sd_detection_model.evaluate_model(X_test_updated, y_test)
            print(accuracy)

            # output_file_name = str(patient_file_name) + '_confusion_matrix.jpg' output_dir = Path( '../../data/' +
            # str(data_type) + '_result/model_' + str(model_no) + '_data_param_set_' + str(dp_set_no))
            # output_dir.mkdir(parents=True, exist_ok=True) conf_matrix_file_name = output_dir / output_file_name
            # sd_detection_model.predict_sd(X_test, y_test, (j + 1), "all_test_patient", "test", (i + 1))

            prediction_file_name = fmu.get_result_full_file_name(patient_file_name='all_test_patient',
                                                                 file_type=fc.PREDICTION_CSV)
            conf_matrix_file_name = fmu.get_result_full_file_name(patient_file_name='all_test_patient',
                                                                  file_type=fc.CONF_MATRIX_JPG)

            sd_detection_model.predict_sd(X_test_updated, y_test, prediction_file_name, conf_matrix_file_name)


if __name__ == "__main__":
    main()
