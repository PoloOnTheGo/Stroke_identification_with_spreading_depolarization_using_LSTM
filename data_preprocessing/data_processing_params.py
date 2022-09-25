class DataProcessingParams:
    def __init__(self, before_sd_buffer, after_sd_buffer, time_window, time_window_shift):
        self.before_sd_buffer = before_sd_buffer
        self.after_sd_buffer = after_sd_buffer
        self.time_window = time_window
        self.time_window_shift = time_window_shift

