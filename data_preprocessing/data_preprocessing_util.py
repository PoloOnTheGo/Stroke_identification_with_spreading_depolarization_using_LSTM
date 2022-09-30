import warnings

import numpy as np

from data_preprocessing import data_constants as dc
from model_creation import model_constants as mc
from data_preprocessing.data_processing_params import DataProcessingParams
from file_management import file_management_constant as fc
from file_management.file_management_util import FileManagementUtil

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def get_corrected_long_sd_blocks(stale_sectioned_sd_block):
    fixed_sectioned_sd_block = pd.DataFrame()
    subset_g = stale_sectioned_sd_block
    subset_g[dc.COLUMN_TEST] = (stale_sectioned_sd_block.iloc[:, :-1].values == dc.VALUE_CTRL).all(1)
    for k, h in subset_g.groupby([(subset_g.test != subset_g.test.shift()).cumsum()]):
        if True in h[dc.COLUMN_TEST].unique() and len(h) > 10:
            h = h.replace({dc.VALUE_SD: dc.VALUE_NOT_SD})
        h = h.drop([dc.COLUMN_TEST], axis=1)
        if k == 1:
            fixed_sectioned_sd_block = h
        else:
            fixed_sectioned_sd_block = fixed_sectioned_sd_block.append(h)
    return fixed_sectioned_sd_block


def get_patient_file_with_corrected_sd_blocks(patient_file):
    fixed_patient_file = ""
    for i, g in patient_file.groupby([(patient_file.mSD != patient_file.mSD.shift()).cumsum()]):
        if dc.VALUE_SD in g[dc.COLUMN_MSD].unique():
            if dc.VALUE_DEPR not in g.iloc[:, :-1].values:
                g = g.replace({dc.VALUE_SD: dc.VALUE_NOT_SD})
            elif (g.iloc[:, :-1].values == dc.VALUE_DEPR).sum() == 1:  # if there is only 1 depr over the whole SD block
                g = g.replace({dc.VALUE_SD: dc.VALUE_NOT_SD})
            elif len(g) > 61:
                g = get_corrected_long_sd_blocks(g)
        if i == 1:
            fixed_patient_file = g
        else:
            fixed_patient_file = fixed_patient_file.append(g)
    return fixed_patient_file


def add_previous_block_part(block, i, j, fixed_patient_file_new, before_sd_buffer):
    if i == 0:
        return fixed_patient_file_new, j
    pre_block = block[i - 1]
    if len(pre_block) >= before_sd_buffer:
        part_from_pre_block = pre_block[-before_sd_buffer:]
    else:
        part_from_pre_block = pre_block

    fixed_patient_file_new = fixed_patient_file_new.append(part_from_pre_block)
    j += 1
    return fixed_patient_file_new, j


def add_post_block_part(block, i, j, fixed_patient_file_new, after_sd_buffer):
    if i == len(block) - 1:
        return fixed_patient_file_new, j
    post_block = block[i]
    if len(post_block) >= after_sd_buffer:
        part_from_post_block = post_block[:after_sd_buffer]
    else:
        part_from_post_block = post_block

    fixed_patient_file_new = fixed_patient_file_new.append(part_from_post_block)
    j += 1
    return fixed_patient_file_new, j


def get_patient_data_only_around_sds(patient_file, before_sd_buffer, after_sd_buffer):
    updated_patient_data = ""
    j = 0
    grouped = patient_file.groupby([(patient_file.mSD != patient_file.mSD.shift()).cumsum()])
    block_list = [g for i, g in grouped]
    for i, g in enumerate(block_list):
        if j == 0:
            start_idx = int(len(g) / 3)
            end_idx = 2 * int(len(g) / 3)
            updated_patient_data = g[start_idx:end_idx]
            j += 1
        if 1 in g[dc.COLUMN_MSD].unique():
            updated_patient_data, j = add_previous_block_part(block_list, i, j, updated_patient_data, before_sd_buffer)
            updated_patient_data = updated_patient_data.append(g)
            j += 1
            updated_patient_data, j = add_post_block_part(block_list, i, j, updated_patient_data, after_sd_buffer)
    return updated_patient_data


def label_encoding(patient_file):
    # cannot use Label_Encoder as few patients does not have SD at all and thus classes gets wrongly labelled
    patient = patient_file.replace({dc.VALUE_CTRL: 0, dc.VALUE_DEPR: 1, dc.VALUE_NOT_SD: 0, dc.VALUE_SD: 1})
    return patient


def split_patient_data_by_time_gap(patient_file):
    split_patient_data = []
    patient_file[dc.COLUMN_SIG_START] = pd.to_datetime(patient_file.index)
    shifted_time = patient_file[dc.COLUMN_SIG_START].diff(periods=1)
    group_samples = (shifted_time.dt.total_seconds() > 61).cumsum()
    grouped = patient_file.groupby(group_samples)
    group_list = [g for k, g in grouped]
    for df_i in group_list:
        df_i = df_i.drop([dc.COLUMN_SIG_START], axis=1)
        split_patient_data.append(df_i)
    return split_patient_data


def split_sequences(time_gap_split_patient_data, time_window, time_window_shift):
    X_patient, y_patient, = [], []
    j = 0
    for sequences in time_gap_split_patient_data:
        for i in range(0, len(sequences), time_window_shift):
            # find the end of this pattern
            end_ix = i + time_window
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences.iloc[i:end_ix, :-1], sequences.iloc[end_ix - 1, -1]
            X_patient.append(seq_x)
            y_patient.append(seq_y)
        j = j + 1
    return X_patient, y_patient


class DataPreprocessingUtil:
    def __init__(self, folder_name=None, data_processing_params: DataProcessingParams = None):
        self.folder_name = folder_name
        self.before_sd_buffer = data_processing_params.before_sd_buffer
        self.after_sd_buffer = data_processing_params.after_sd_buffer
        self.time_window = data_processing_params.time_window
        self.time_window_shift = data_processing_params.time_window_shift

    def read_data_for_each_patient_in_list(self):
        files = []
        # Create a dataframe list by using a list comprehension
        _, file_full_paths = FileManagementUtil(folder_path=self.folder_name).get_all_files_in_directory()
        for file_full_path in file_full_paths:
            files.append(pd.read_csv(file_full_path, index_col=dc.COLUMN_SIG_START, parse_dates=True))
        return files

    def load_individual_patient_data(self, patient_file):
        patient_file_updated = get_patient_file_with_corrected_sd_blocks(patient_file)
        patient_file_updated = label_encoding(patient_file_updated)
        if fc.TRAIN_DATA_PATH in self.folder_name:
            patient_file_updated = get_patient_data_only_around_sds(patient_file_updated, self.before_sd_buffer,
                                                                    self.after_sd_buffer)
        time_gap_split_patient_data = split_patient_data_by_time_gap(patient_file_updated)
        X_patient, y_patient = split_sequences(time_gap_split_patient_data, self.time_window,
                                               self.time_window_shift)
        return X_patient, y_patient

    def load_preprocessed_dataset(self):
        patient_files = self.read_data_for_each_patient_in_list()
        X, y = [], []

        print('------------------------------------------')
        print(self.folder_name)
        for patient_file in patient_files:
            X_patient, y_patient = self.load_individual_patient_data(patient_file)

            X.extend(X_patient)
            y.extend(y_patient)
        X, y = np.array(X), np.array(y)
        y = y.reshape(len(y), 1)
        print(X.shape, y.shape)
        return X, y

    def get_input_shape(self, mp):
        in_shape = ()
        if mp.model_type == mc.SIMPLE_LSTM:
            in_shape = (self.time_window, dc.NO_OF_COLUMNS)
        elif mp.model_type == mc.CNN_LSTM:
            in_shape = (None, mp.clh.n_length, dc.NO_OF_COLUMNS)
        elif mp.model_type == mc.CONV_LSTM:
            in_shape = (mp.clh.n_steps, 1, mp.clh.n_length, dc.NO_OF_COLUMNS)
        return in_shape
