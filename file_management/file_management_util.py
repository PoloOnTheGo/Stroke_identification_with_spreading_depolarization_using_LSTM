from pathlib import Path
from os import listdir
from os.path import isfile, join  # for combining multiple files into one dataframe
from file_management import file_management_constant as fmc


class FileManagementUtil:
    def __init__(self, folder_path=None):
        self.folder_path = folder_path

    def save_json_file(self, file_name, obj_json):
        full_file_path = self.path_creator(file_name)
        with open(full_file_path, "w") as json_file:
            json_file.write(obj_json)

    def path_creator(self, file_name):
        output_dir = Path(self.folder_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        full_file_path = output_dir / file_name
        return full_file_path

    def get_all_files_in_directory(self):
        files = []
        # Create a dataframe list by using a list comprehension
        for file in sorted(listdir(self.folder_path)):
            full_file_path = join(self.folder_path, file)
            if isfile(full_file_path):
                files.append(full_file_path)
        return files

    def get_result_full_file_name(self, patient_file_name=None, file_type=None):
        file_name = file_type.format(patient_file_name=patient_file_name)
        fullname = self.path_creator(file_name)
        print(fullname)
        return fullname

    def set_result_full_path(self, data_type=None, dp_set_no=None, model_no=None):
        self.folder_path = fmc.RESULT_PATH.format(data_type=data_type, data_param_set=dp_set_no, model_set=model_no)

