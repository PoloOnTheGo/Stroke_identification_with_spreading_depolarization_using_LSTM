import json

from file_management.file_management_util import FileManagementUtil


class DataProcessingParams:
    def __init__(self,  time_window, time_window_shift, before_sd_buffer=None, after_sd_buffer=None):
        self.before_sd_buffer = before_sd_buffer
        self.after_sd_buffer = after_sd_buffer
        self.time_window = time_window
        self.time_window_shift = time_window_shift

    def save_data_params(self, fmu: FileManagementUtil):
        fmu.save_json_file('data_params.json', json.dumps(self, default=lambda o: o.__dict__, indent=4))
