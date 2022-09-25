from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten, LSTM, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.metrics import BinaryAccuracy
from keras.metrics import FalseNegatives
from keras.models import Sequential
from sklearn import metrics

from model_creation import model_constants as mc
from model_creation.hyper_parameters.model_hyper_parameters import ModelHyperParameters


def simple_lstm_model(model, mh, input_shape):
    no_of_layers_and_filters = mh.llh.no_of_layers_and_filters
    for i in range(len(no_of_layers_and_filters) - 1):
        model.add(LSTM(no_of_layers_and_filters[i], return_sequences=True, input_shape=input_shape))
    model.add(LSTM(no_of_layers_and_filters[-1]))
    model.add(mh.llh.dropout)

    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[BinaryAccuracy()])
    # model.add(Dense(mh.dlh.no_of_neurons, activation=mh.dlh.act_func))
    # model.add(Dense(mc.N_OUTPUT, activation=mc.ACTIVATION_SIGMOID))
    # # model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER,
    # #               metrics=[BinaryAccuracy(), FalseNegatives()])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[BinaryAccuracy()])


def cnn_lstm_model(model, mh, input_shape):
    cnn_n = mh.clh.no_of_layers_and_filters
    for i in range(len(cnn_n) - 1):
        model.add(TimeDistributed(Conv1D(filters=cnn_n[i], kernel_size=mh.clh.kernel_size,
                                         activation=mc.ACTIVATION_RELU), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=cnn_n[-1], kernel_size=mh.clh.kernel_size, activation=mc.ACTIVATION_RELU)))
    model.add(TimeDistributed(mh.clh.dropout))
    model.add(TimeDistributed(mh.clh.pooling))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(mh.llh.no_of_layers_and_filters[-1]))
    model.add(mh.llh.dropout)
    model.add(Dense(mh.dlh.no_of_neurons, activation=mh.dlh.act_func))
    model.add(Dense(mc.N_OUTPUT, activation=mc.ACTIVATION_SIGMOID))
    model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER,
                  metrics=[BinaryAccuracy(), FalseNegatives()])


def conv_lstm_model(model, mh, input_shape):
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
    model.compile(loss=mc.BINARY_CROSSENTROPY_LOSS, optimizer=mc.ADAM_OPTIMIZER, metrics=[BinaryAccuracy()])


def save_prediction(X_for_prediction, y_for_prediction, predicted_y, model_no, patient_file_name, data_type):
    df_result = pd.DataFrame(data=[[x[-1]] for x in X_for_prediction])
    df_result['Test_y'] = np.reshape(y_for_prediction, len(y_for_prediction))
    df_result['Predicted_y'] = np.reshape(predicted_y, len(predicted_y))

    output_file_name = str(patient_file_name) + '_prediction.csv'
    output_dir = Path('../../data/' + str(data_type) + '_result/' + str(model_no))
    output_dir.mkdir(parents=True, exist_ok=True)
    fullname = output_dir / output_file_name  # os.path.join(output_dir, output_file_name)
    print(fullname)
    df_result.to_csv(fullname)


def save_confusion_matrix(y_for_prediction, predicted_y, model_no, patient_file_name, data_type):
    confusion_matrix = metrics.confusion_matrix(y_for_prediction, predicted_y)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.title(get_metrics_vales(confusion_matrix, y_for_prediction, predicted_y))
    output_file_name = str(patient_file_name) + 'confusion_matrix.jpg'
    output_dir = Path('../../data/' + str(data_type) + '_result/' + str(model_no))
    output_dir.mkdir(parents=True, exist_ok=True)
    fullname = output_dir / output_file_name
    plt.savefig(fullname)
    plt.show()


def get_metrics_vales(confusion_matrix, y_for_prediction, predicted_y):
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    actual = y_for_prediction
    # Overall accuracy [ACC = (TP+TN)/(TP+FP+FN+TN)]
    Accuracy = metrics.accuracy_score(actual, predicted_y)
    # Precision or positive predictive value [PPV = TP/(TP+FP)]
    Precision = metrics.precision_score(actual, predicted_y)
    # Sensitivity, hit rate, recall, or true positive rate [TPR = TP/(TP+FN)]
    Sensitivity_recall = metrics.recall_score(actual, predicted_y)
    # Specificity or true negative rate [TNR = TN/(TN+FP)]
    Specificity = metrics.recall_score(actual, predicted_y, pos_label=0)
    F1_score = metrics.f1_score(actual, predicted_y)
    # Negative predictive value [NPV = TN/(TN+FN)]
    # Fall out or false positive rate [FPR = FP/(FP+TN)]
    # False negative rate [FNR = FN/(TP+FN)]
    false_negative_rate = FN / (TP + FN)
    print("false_negative_rate:" + str(false_negative_rate))
    # False discovery rate [FDR = FP/(TP+FP)]
    # metrics:
    result = {"Accuracy": Accuracy, "Precision": Precision, "Sensitivity_recall": Sensitivity_recall,
              "Specificity": Specificity, "F1_score": F1_score}
    return str(result)


class SdDetectionModel(Sequential):
    def __init__(self, model_hyper_parameters: ModelHyperParameters, input_shape):
        super().__init__()
        self.hyper_parameters = model_hyper_parameters
        model_type = self.hyper_parameters.model_type
        # Initialising the CNN
        if model_type == mc.SIMPLE_LSTM:
            simple_lstm_model(self, self.hyper_parameters, input_shape)
            print('model created')

        elif model_type == mc.CNN_LSTM:
            cnn_lstm_model(self, self.hyper_parameters, input_shape)
            print('model created')

        elif model_type == mc.CONV_LSTM:
            conv_lstm_model(self, self.hyper_parameters, input_shape)
            print('model created')

    def save_model(self):
        model_json = self.to_json()
        with open("../model-bw.json", "w") as json_file:
            json_file.write(model_json)
        self.save_weights('../model-bw.h5')

    def train_model(self, train_X, train_y):
        # fit network
        self.fit(train_X, train_y, epochs=self.hyper_parameters.epochs, batch_size=self.hyper_parameters.batch_size,
                 verbose=self.hyper_parameters.verbose)
        print('model fitted')

    def evaluate_model(self, test_X, test_y):
        loss, accuracy = self.evaluate(test_X, test_y, batch_size=self.hyper_parameters.batch_size,
                                       verbose=self.hyper_parameters.verbose)
        return loss, accuracy

    def predict_sd(self, X_for_prediction, y_for_prediction, model_no, patient_file_name, data_type):
        # evaluate model
        predicted_y = (self.predict(X_for_prediction, batch_size=self.hyper_parameters.batch_size,
                                    verbose=0) > self.hyper_parameters.threshold).astype("int32")

        save_prediction(X_for_prediction, y_for_prediction, predicted_y, model_no, patient_file_name, data_type)
        save_confusion_matrix(y_for_prediction, predicted_y, model_no, patient_file_name, data_type)
