import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, LSTM, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.metrics import BinaryAccuracy
from keras.metrics import FalseNegatives
from keras.models import Sequential
from sklearn import metrics

from data_preprocessing import data_constants as dc
from file_management.file_management_util import FileManagementUtil
from model_creation import model_constants as mc


def simple_lstm_model(mh, input_shape):
    model = Sequential()
    no_of_layers_and_filters = mh.llh.no_of_layers_and_filters
    for i in range(len(no_of_layers_and_filters) - 1):
        model.add(LSTM(no_of_layers_and_filters[i], return_sequences=True, input_shape=input_shape))
    model.add(LSTM(no_of_layers_and_filters[-1]))
    model.add(mh.llh.dropout)

    model.add(Dense(mh.dlh.no_of_neurons, activation=mh.dlh.act_func))
    model.add(Dense(mc.N_OUTPUT, activation=mc.ACTIVATION_SIGMOID))
    model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER,
                  metrics=[BinaryAccuracy(), FalseNegatives()])
    return model


def cnn_lstm_model(mh, input_shape):
    model = Sequential()
    cnn_n = mh.clh.no_of_layers_and_neurons
    for i in range(len(cnn_n) - 1):
        model.add(TimeDistributed(Conv1D(filters=cnn_n[i], kernel_size=mh.clh.kernel_size,
                                         activation=mc.ACTIVATION_RELU), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=cnn_n[-1], kernel_size=mh.clh.kernel_size, activation=mc.ACTIVATION_RELU)))
    model.add(TimeDistributed(mh.clh.dropout))
    model.add(TimeDistributed(mh.clh.pooling))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(mh.llh.no_of_layers_and_neurons[-1]))
    model.add(mh.llh.dropout)
    model.add(Dense(mh.dlh.no_of_neurons, activation=mh.dlh.act_func))
    model.add(Dense(mc.N_OUTPUT, activation=mc.ACTIVATION_SIGMOID))
    model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER,
                  metrics=[BinaryAccuracy(), FalseNegatives()])
    return model


def conv_lstm_model(mh, input_shape):
    model = Sequential()
    mh.clh = mh.clh
    conv_n = mh.clh.no_of_layers_and_filters
    for i in range(len(conv_n) - 1):
        model.add(ConvLSTM2D(filters=conv_n[i], kernel_size=mh.clh.kernel_size, activation=mc.ACTIVATION_RELU,
                             return_sequences=True, input_shape=input_shape))
    model.add(ConvLSTM2D(filters=conv_n[-1], kernel_size=mh.clh.kernel_size,
                         activation=mc.ACTIVATION_RELU, input_shape=input_shape))
    model.add(mh.clh.dropout)
    model.add(Flatten())
    model.add(Dense(mh.dlh.no_of_neurons, activation=mh.dlh.act_func))
    model.add(Dense(mc.N_OUTPUT, activation=mc.ACTIVATION_SIGMOID))
    model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER,
                  metrics=[BinaryAccuracy(), FalseNegatives()])
    return model


def save_prediction(X_pred, y_for_prediction, predicted_y, prediction_file_name):
    df_result = pd.DataFrame(data=[[x[-1]] for x in X_pred])
    df_result['Test_y'] = np.reshape(y_for_prediction, len(y_for_prediction))
    df_result['Predicted_y'] = np.reshape(predicted_y, len(predicted_y))
    df_result.to_csv(prediction_file_name)


def save_confusion_matrix(y_for_prediction, predicted_y, conf_matrix_file_name):
    confusion_matrix = metrics.confusion_matrix(y_for_prediction, predicted_y)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.rcParams.update({'font.size': 8})
    plt.title(get_metrics_vales(confusion_matrix, y_for_prediction, predicted_y))
    plt.savefig(conf_matrix_file_name)
    plt.close()
    # plt.show()


def get_metrics_vales(confusion_matrix, y_for_prediction, predicted_y):
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    actual = y_for_prediction
    # Overall accuracy [ACC = (TP+TN)/(TP+FP+FN+TN)]
    accuracy = metrics.accuracy_score(actual, predicted_y)
    # Precision or positive predictive value [PPV = TP/(TP+FP)]
    precision = metrics.precision_score(actual, predicted_y)
    # Sensitivity, hit rate, recall, or true positive rate [TPR = TP/(TP+FN)]
    sensitivity_recall = metrics.recall_score(actual, predicted_y)
    # Specificity or true negative rate [TNR = TN/(TN+FP)]
    specificity = metrics.recall_score(actual, predicted_y, pos_label=0)
    f1_score = metrics.f1_score(actual, predicted_y)
    # Negative predictive value [NPV = TN/(TN+FN)]
    # Fall out or false positive rate [FPR = FP/(FP+TN)]
    # False negative rate [FNR = FN/(TP+FN)]
    false_negative_rate = FN / (TP + FN)
    print("false_negative_rate:" + str(false_negative_rate))
    # False discovery rate [FDR = FP/(TP+FP)]
    # metrics:
    result = 'Accuracy : ' + str(accuracy) + ', Precision : ' + str(precision) \
             + ',\nSensitivity_recall : ' + str(sensitivity_recall) + ', Specificity : ' + str(specificity) \
             + ',\nFalse_negative : ' + str(false_negative_rate) + ', F1_score : ' + str(f1_score)

    return str(result)


def reshape_X_for_model(mp, X):
    if mp.model_type == mc.SIMPLE_LSTM:
        print("No reshape required")
    elif mp.model_type == mc.CNN_LSTM:
        X = X.reshape((X.shape[0], mp.clh.n_steps, mp.clh.n_length, dc.NO_OF_COLUMNS))
    elif mp.model_type == mc.CONV_LSTM:
        X = X.reshape((X.shape[0], mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS))
    return X


def save_model(model, fmu: FileManagementUtil):
    fmu.save_json_file(file_name='model.json', obj_json=model.to_json())
    weights_file_path = fmu.path_creator(file_name='model.h5')
    model.save_weights(weights_file_path)


def train_model(model, mp, train_X, train_y):
    # fit network
    model.fit(train_X, train_y, epochs=mp.epochs, batch_size=mp.batch_size, verbose=mp.verbose)
    print('model fitted')


def predict_sd(model, mp, X_pred, X_updated_for_pred, y_for_prediction, prediction_file_name, conf_matrix_file_name):
    # evaluate model
    predicted_y = (model.predict(X_updated_for_pred, batch_size=mp.batch_size, verbose=0) > mp.threshold).astype(
        "int32")

    save_prediction(X_pred, y_for_prediction, predicted_y, prediction_file_name)
    save_confusion_matrix(y_for_prediction, predicted_y, conf_matrix_file_name)


def create_model(model_hyper_parameters, input_shape):
    model_type = model_hyper_parameters.model_type
    model = None
    # Initialising the CNN
    if model_type == mc.SIMPLE_LSTM:
        model = simple_lstm_model(model_hyper_parameters, input_shape)
        print('model created')

    elif model_type == mc.CNN_LSTM:
        model = cnn_lstm_model(model_hyper_parameters, input_shape)
        print('model created')

    elif model_type == mc.CONV_LSTM:
        model = conv_lstm_model(model_hyper_parameters, input_shape)
        print('model created')
    return model
