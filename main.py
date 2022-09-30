import warnings

from keras.layers import MaxPooling1D, Dropout
from keras.models import model_from_json
import numpy as np

from data_preprocessing.data_preprocessing_util import DataPreprocessingUtil
from data_preprocessing.data_processing_params import DataProcessingParams
from data_preprocessing import data_constants as dc
from model_creation import model_constants as mc
from file_management import file_management_constant as fc

from file_management.file_management_util import FileManagementUtil

import json
from types import SimpleNamespace

from model_creation import model_util
from model_creation.hyper_parameters.conv_layer_hyper_parameters import ConvLayerHyperparameter
from model_creation.hyper_parameters.dense_layer_hyper_parameters import DenseLayerHyperparameter
from model_creation.hyper_parameters.lstm_layer_hyper_parameters import LstmLayerHyperparameter
from model_creation.hyper_parameters.model_hyper_parameters import ModelHyperParameters

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def load_model(fmu: FileManagementUtil):
    model_json = fmu.load_json_file(file_name='model.json')
    model = model_from_json(model_json)
    # load weights into new model
    weights_file_path = fmu.path_creator(file_name='model.h5')
    model.load_weights(weights_file_path)
    print("Loaded model from disk")
    return model


def load_data_params(fmu: FileManagementUtil):
    data_params_json = fmu.load_json_file(file_name='data_params.json')
    data_params = json.loads(data_params_json, object_hook=lambda d: SimpleNamespace(**d))
    return data_params


def predict_validation_patient_files(model, data_params: DataProcessingParams, model_params):
    validation_data_folder = fc.DATA_ROOT_PATH_MAIN + fc.VALIDATION_DATA_PATH
    validation_patient_file_names, validation_patient_file_paths = FileManagementUtil(validation_data_folder).get_all_files_in_directory()
    dpu = DataPreprocessingUtil(validation_data_folder, data_params)
    # validation_patient_file_names = dpu.read_data_for_each_patient_in_list()

    fmu = FileManagementUtil()
    fmu.set_result_full_path(root=fc.DATA_ROOT_PATH_MAIN, data_type=fc.VALIDATION_DATA_PATH, dp_set_no='4',
                             model_no='11')
    for i in range(len(validation_patient_file_paths)):
        file_name = validation_patient_file_names[i]
        file_name_w_o_ext = ''.join(file_name.split())[:-4]
        print(str(file_name_w_o_ext))
        patient_file = pd.read_csv(validation_patient_file_paths[i], index_col=dc.COLUMN_SIG_START, parse_dates=True)
        X_val, y_val = dpu.load_individual_patient_data(patient_file)

        X_val, y_val = np.array(X_val), np.array(y_val)
        y_val = y_val.reshape(len(y_val), 1)
        print(X_val.shape, y_val.shape)

        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)

        X_val_updated = model_util.reshape_X_for_model(model_params, X_val)

        prediction_file_name = fmu.get_result_full_file_name(patient_file_name=file_name_w_o_ext,
                                                             file_type=fc.PREDICTION_CSV)
        conf_matrix_file_name = fmu.get_result_full_file_name(patient_file_name=file_name_w_o_ext,
                                                              file_type=fc.CONF_MATRIX_JPG)

        model_util.predict_sd(model=model, mp=model_params, X_pred=X_val, X_updated_for_pred=X_val_updated,
                              y_for_prediction=y_val, prediction_file_name=prediction_file_name,
                              conf_matrix_file_name=conf_matrix_file_name)


def get_cnn_lstm_model_param(n_layers, dropout, n_steps, n_length):
    clh = ConvLayerHyperparameter(act_func=mc.ACTIVATION_RELU, kernel_size=2, no_of_layers_and_neurons=n_layers,
                                  dropout=dropout, pooling=MaxPooling1D(pool_size=2), n_steps=n_steps,
                                  n_length=n_length)
    llh = LstmLayerHyperparameter(no_of_layers_and_neurons=[100], dropout=Dropout(0.5))
    dlh = DenseLayerHyperparameter(no_of_neurons=100, act_func=mc.ACTIVATION_RELU)
    model_hp = ModelHyperParameters(model_type=mc.CNN_LSTM, threshold=0.5, clh=clh, llh=llh, dlh=dlh, verbose=1,
                                    batch_size=64, epochs=70)
    return model_hp


def run_experiment():
    fmu = FileManagementUtil(fc.MODEL_PATH)
    model = load_model(fmu)
    model_params = get_cnn_lstm_model_param(n_layers=[64, 64], dropout=Dropout(0.5), n_steps=2, n_length=16)
    data_params = load_data_params(fmu)
    predict_validation_patient_files(model, data_params, model_params)


if __name__ == "__main__":
    run_experiment()
