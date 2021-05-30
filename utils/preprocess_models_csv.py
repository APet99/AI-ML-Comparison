import csv
import os
from pathlib import Path
from utils.prep_data import *
from utils.results import get_repository_root_folder_path

RESULTS_PATH = get_repository_root_folder_path().joinpath("results").joinpath("grayscale")
'''Path.cwd() / "results" / "grayscale"''' #TODO Update RESULTS_PATH when merge is complete with Main/Models

def convert_dictionary_to_dataframe(model_dictionary: dict):
    print(model_dictionary)
    tmp_dict = {}

    df = pd.DataFrame(tmp_dict, columns=["Time", "Is Correct?"])
    return df


def load_model_generation():
    directory_paths = []
    for i in os.listdir(RESULTS_PATH):
        dir_path = os.path.join(RESULTS_PATH, i)
        if os.path.isdir(dir_path):
            directory_paths.append(dir_path)

    if len(directory_paths) < 1:
        print("ERROR! No directories found in results path. (File: preprocess_models_csv.py)")
        print("Did you set your RESULTS_PATH in preprocess_models_csv.py?")
        return None

    file_paths = []
    for i in directory_paths:
        for root, dirs, files in os.walk(i):
            for file in files:
                if file.startswith('prediction'):
                    file_paths.append(root + '\\' + str(file))

    column_names = []
    data = []
    for tmp_path in file_paths:
        with open(str(tmp_path)) as csv_file:
            csv_data = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_data:
                data_dict = {}
                if line_count == 0:
                    column_names = row
                    line_count = line_count + 1
                else:
                    for i in range(len(column_names)):
                        try:
                            data_dict[column_names[i]] = row[i]
                        except Exception:
                            print("ERROR! Data Processing Error in CSV File (File: preprocess_models_csv.py)")
                    data.append(data_dict)
                    line_count = line_count + 1

    model_names = []
    model_dict = {}

    for dict in data:
        model_name = dict.get('model_name')
        if model_name not in model_names:
            model_names.append(model_name)
            model_dict[model_name] = []

        try:
            tmp_pred = Prediction(float(dict.get('time')), dict.get('is_correct'))
            tmp_array = model_dict[model_name]
            tmp_array.append(tmp_pred)
        except Exception:
            print("ERROR! Passing a non-integer or non-Boolean into Prediction. (File: preprocess_models_csv.py)")

    return model_dict


def load_model_timings():
    score_path = RESULTS_PATH.joinpath('score_results_0.csv')
    if not score_path.is_file():
        print("ERROR! No score results found in results path. (File: preprocess_models_csv.py)")
        print("Did you set your RESULTS_PATH in preprocess_models_csv.py?")
        return None

    file_paths = []
    for i in os.listdir(RESULTS_PATH):
        if os.path.isfile(os.path.join(RESULTS_PATH, i)):
            file_paths.append(os.path.join(RESULTS_PATH, i))

    column_names = []
    score_results = []
    for tmp_path in file_paths:
        with open(str(tmp_path)) as csv_file:
            csv_data = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_data:
                data_dict = {}
                if line_count == 0:
                    column_names = row
                    line_count = line_count + 1
                else:
                    for i in range(len(column_names)):
                        try:
                            data_dict[column_names[i]] = row[i]
                        except Exception:
                            print("ERROR! Data Processing Error in CSV File (File: preprocess_models_csv.py)")
                    score_results.append(data_dict)
                    line_count = line_count + 1

    model_dict = {}
    model_names = []

    for score_dict in score_results:
        model_name = score_dict.get('model_name')
        if model_name not in model_names:
            model_names.append(model_name)
            model_dict[model_name] = []

        try:
            tmp_score = Score(float(score_dict.get("average_accuracy")), float(score_dict.get("time_to_score")))
            tmp_array = model_dict[model_name]
            tmp_array.append(tmp_score)
        except Exception:
            print("ERROR! Passing a non-integer into Score. (File: preprocess_models_csv.py)")

    return model_dict


class Prediction:

    def __init__(self, time, is_correct):
        self.time = time
        self.is_correct = is_correct


class Score:

    def __init__(self, accuracy, time):
        self.accuracy = accuracy
        self.time = time
