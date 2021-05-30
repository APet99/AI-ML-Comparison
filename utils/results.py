import atexit
import csv
import os
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydrive.files import GoogleDriveFile

from utils.gdrive_utils.gdrive_utils import get_repository_root_folder_path, get_google_drive_file
from utils.hardware_specs import create_system_info_json
from utils.logger import log


def _get_results_folder_path() -> Path:
    return get_repository_root_folder_path().joinpath('results')


def _get_google_drive_results_folder():
    return get_google_drive_file(file_name='results', is_file_a_folder=True)


class CSVResult:
    def __init__(self, column_names: list, path):
        """
        initializes a CSVResult object

        :param column_names: list of column names as strings
        :param path: relative path to the results folder specified in ResultsSaver
        column names and values being values associated with that column
        """
        self.column_names = column_names
        self.__results = []  # {[{'column_1': 121, 'column_2': 0}, {'column_1': 120, 'column_2': 2}, ...]}
        self.path = ResultSaver.results_folder_path.joinpath(path)  # relative to local results

    def add_result(self, result: dict):
        """
        adds a result to result field, must have the same columns as in self.column_names

        :param result: dictionary of column keys mapped to values of the column
        :throws Exception: if the result keys are not the same as self.column_names
        :return: adds a result to self.results
        """
        if len(result.keys()) != len(self.column_names):
            raise Exception(f'All keys must be declared within column_names:{os.linesep}',
                            f'length of result.keys() != self.column_names{os.linesep}',
                            f'{len(result.keys())} != {len(self.column_names)}')

        for result_key in result.keys():
            if result_key not in self.column_names:
                raise Exception(
                    f'All keys must be declared within column_names: {result_key} is not in {result.keys()}')
        self.__results.append(result)

    def save_results(self):
        """
        Saves self.results to the location of self.path
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        csv_file_path = str(self.path)
        with open(csv_file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['id', *self.column_names], lineterminator='\n')
            writer.writeheader()
            for i, result in enumerate(self.__results):
                if 'id' not in result.keys():
                    result.update({'id': i})
                writer.writerow(result)

    def plot(self, x_coordinate: str, y_coordinate: str, title=None, type_='scatter', y2_coordinate: str = None):
        if x_coordinate not in self.column_names:
            raise Exception(
                f'Coordinate must be declared within column_names: {x_coordinate} is not in {self.column_names}')
        if y_coordinate not in self.column_names:
            raise Exception(
                f'Coordinate must be declared within column_names: {y_coordinate} is not in {self.column_names}')

        if y2_coordinate:
            if y2_coordinate not in self.column_names:
                raise Exception(
                    f'Coordinate must be declared within column_names: {y2_coordinate} is not in {self.column_names}')
            if y_coordinate == y2_coordinate:
                raise Exception(
                    f'Y Coordinates must not be the same: y_coordinate={y_coordinate} '
                    f'and y2_coordinate={y2_coordinate}')

        log(f'plotting {self.path.relative_to(get_repository_root_folder_path())}')

        fig = plt.figure()
        if not title:
            plt.title(str(self.path.name))
        else:
            plt.title(str(title))

        plt.xlabel(x_coordinate)
        plt.ylabel(y_coordinate)

        if type_ == 'scatter':
            fig.set_size_inches(15, 10)
            plt.scatter([x[x_coordinate] for x in self.__results], [y[y_coordinate] for y in self.__results])

        elif type_ == 'bar':
            fig.set_size_inches(24, 10.5)
            plt.barh([x[x_coordinate] for x in self.__results], [y[y_coordinate] for y in self.__results])

            plt.tick_params(axis='y', labelsize=8)
            # plt.xticks(rotation=-15)

        else:
            if not y2_coordinate:
                raise Exception(f'"y2_coordinate" must be set if "type_" is not "scatter" or "bar" ')
            other_axis = plt.twinx()
            other_axis.plot([x[x_coordinate] for x in self.__results], [y[y2_coordinate] for y in self.__results])
            plt.scatter([x[x_coordinate] for x in self.__results], [y[y_coordinate] for y in self.__results])

        self.path.parent.mkdir(exist_ok=True, parents=True)

        fig.savefig(self.path.parent.joinpath(self.path.name.split('.')[0] + ".png"), dpi=1000)
        plt.cla()
        plt.clf()
        plt.close()  # When printing images from multiple algo in succession, closing the plot MIGHT have side affects


class ResultSaver:
    """
    Handles the saving of all specified results and uploads the results folder
    to the ItsClassified Google Drive if specified.
    """
    results_folder_path = _get_results_folder_path()  # absolute path of folder where the Results will be saved
    _gdrive_results_folder = _get_google_drive_results_folder()  # google drive results folder object
    _folder_link = None  # link to newly uploaded folder

    UPLOAD_RESULTS = False  # if set to True will attempt to upload results folder to google drive
    RESULTS_TO_SAVE = []  # list of all the CSVResults to save
    DATE_TIME = datetime.now().strftime('%c')

    @staticmethod
    def generate_file_path_token_document():
        try:
            file = open(ResultSaver.results_folder_path.joinpath("results_path.dog"), "w")
            print(ResultSaver.DATE_TIME)
            file.write(ResultSaver.DATE_TIME)
            file.close()
        except Exception:
            print("ERROR! Generating Watchdog Path Token Document, Failed to Write (File: results.py)")

    @staticmethod
    def upload_results(local_parent_folder_path: Path = results_folder_path,
                       previous_gdrive_parent_folder: GoogleDriveFile = None):
        """
        Recursively uploads the content of the specified folder path to the ItsClassified Google Drive

        :param local_parent_folder_path: absolute path of the local folder to upload
        :param previous_gdrive_parent_folder: if uploaded folder should be a child of a specific google drive folder
        :return: None but the specified local folder and its contents should be uploaded to google drive
        """
        parent_folder_name = local_parent_folder_path.name if local_parent_folder_path.name != 'results' \
            else ResultSaver.DATE_TIME

        for local_child_folder_path in local_parent_folder_path.iterdir():
            gdrive_parent_folder = get_google_drive_file(file_name=parent_folder_name,
                                                         is_file_a_folder=True,
                                                         gdrive_parent_folder=previous_gdrive_parent_folder)
            if not gdrive_parent_folder.get("id"):
                log(f'Uploading "{gdrive_parent_folder["title"]}"')
            gdrive_parent_folder.Upload()

            if not ResultSaver._folder_link:
                ResultSaver._folder_link = gdrive_parent_folder['alternateLink']

            if not local_child_folder_path.is_dir():
                gdrive_child_folder = get_google_drive_file(file_name=local_child_folder_path.name,
                                                            gdrive_parent_folder=gdrive_parent_folder)
                gdrive_child_folder.SetContentFile(str(local_child_folder_path))

                if not gdrive_child_folder.get('id'):
                    log(f'Uploading "{gdrive_parent_folder["title"]}/{gdrive_child_folder["title"]}"')
                gdrive_child_folder.Upload()
            else:
                gdrive_child_folder = get_google_drive_file(file_name=local_child_folder_path.name,
                                                            is_file_a_folder=True,
                                                            gdrive_parent_folder=gdrive_parent_folder)
                if not gdrive_child_folder.get('id'):
                    log(f'Uploading "{gdrive_parent_folder["title"]}/{gdrive_child_folder["title"]}"')
                gdrive_child_folder.Upload()
                ResultSaver.upload_results(previous_gdrive_parent_folder=gdrive_parent_folder,
                                           local_parent_folder_path=local_child_folder_path)

    @staticmethod
    def plot_result(result: CSVResult):
        """
        plots the CSVResult passed with designated parameters

        :param result: CSVResult, filled with data and preferable within the names listed bellow
        :return: Plots the CSV results and saves it in the same directory as the CSV result
        """
        if 'predict' in result.path.name.lower():
            result.plot(x_coordinate='id', y_coordinate='time', type_='scatter')

        elif 'score' in result.path.name.lower():
            result.plot(x_coordinate='model_name', y_coordinate='average_accuracy', type_='bar and line',
                        y2_coordinate='time_to_score')

        elif 'classification' in result.path.name.lower():
            result.plot(x_coordinate='Classification', y_coordinate='precision', type_='bar')

    @staticmethod
    def generate_hardware_specifications(results_path: Path):
        create_system_info_json(results_path)

    @staticmethod
    @atexit.register
    def _save_results():
        """
        Shutdown hook that saves all the results and uploads those results to Google Drive
        """
        results_path = ResultSaver.results_folder_path

        log("Saving results...")
        for result in ResultSaver.RESULTS_TO_SAVE:
            result.save_results()
            ResultSaver.plot_result(result)

        if ResultSaver.UPLOAD_RESULTS:
            results_path.mkdir(exist_ok=True)
            ResultSaver.generate_hardware_specifications(
                results_path)  # Generates a json file to be added to the results directory for hardware specifications
            try:
                log(f"Uploading Results stored in {results_path}...")
                ResultSaver._gdrive_results_folder.Upload()
                user_folder = get_google_drive_file(file_name=Path.home().name,
                                                    gdrive_parent_folder=ResultSaver._gdrive_results_folder,
                                                    is_file_a_folder=True)
                user_folder.Upload()
                ResultSaver.upload_results(results_path, user_folder)
                ResultSaver.generate_file_path_token_document()
                log(f'Link to newly uploaded folder: {ResultSaver._folder_link}')
            except Exception:
                traceback.print_exc()
                log(f"There was an error while trying to upload. {os.linesep} DO NOT DELETE RESULTS IF YOU RUN AGAIN "
                    f"WITH ResultSaver {os.linesep}ACTIVE IT WILL DELETE THE \"results\" FOLDER ")
