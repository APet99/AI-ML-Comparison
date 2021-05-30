"""
Model.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 3/07/2021

Model.py is used to encapsulate all scikit model functionality for abstracted use.
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

from runner import prepare_results_directory
from utils.prep_data import generate_category_labels_from_csv
from utils.serialization import serialize


class Model:
    def __init__(self, model_name: str, model_object, is_color: bool = True):
        self.dataset_type = self.__set_colored_settings(is_color)
        self.location_of_dataset, self.location_of_categories = self.__prepare_locations(is_color)
        self.model_name = model_name
        if model_object is not None:
            self.model_object = model_object
            self.serialized_location = self.__get_model_location()
        else:
            raise ValueError("ERROR: A NEW model object is required.")

    def __get_model_location(self) -> Path:
        return Path(f'models/{self.model_name}_{self.dataset_type}.pickle')

    def fit(self, features_train, labels_train):
        """
        Method which "trains" the models
        :param features_train: A numpy array which represents the images within the training dataset
        :param labels_train: A numpy array which represents the classification labels
        :return: Model
        """
        return self.model_object.fit(features_train, labels_train)

    # model_name
    def predict_dataset(self, features, labels):
        """
        Method which classifies images based on the "training" set
        :param features: A numpy array consisting of the images from the dataset
        :param labels: A numpy array consisting of the labels corresponding to the images
        :return:
        """
        count_right = 0
        count_wrong = 0
        results = []

        for feature, label in zip(features, labels):
            feature_instance = feature.reshape(1, -1)
            predicted_y, time_to_predict = self.predict(feature_instance)

            if predicted_y[0] != label:
                count_wrong += 1
            else:
                count_right += 1
            results.append(
                {'model_name': self.model_name, 'time': time_to_predict, 'predicted': predicted_y[0], 'actual': label,
                 'is_correct': (predicted_y[0] == label)})

        return results

    def predict(self, feature):
        """
        Method where the model attempts to classify the given image from the feature set and returns predicted
        classification id
        :param feature: A numpy array consisting of a single value (image)
        :return: Integer
        """
        return self._timer(self.model_object.predict, feature)

    def score(self, features_test, labels_test):
        """
        Method in which the model predicts the elements within the dataset and returns an average for overall accuracy
        :param features_test: A numpy array which represents the images within the testing dataset
        :param labels_test: A numpy array which represents the classification labels within the testing dataset
        :return: Float representing the average model accuracy across the test data set.
        """
        return self._timer(self.model_object.score, features_test, labels_test)

    def cross_validate(self, features, labels) -> list:
        cv = cross_validate(self.model_object, features, labels)
        length = len(cv.get(list(cv.keys())[0]))
        output = [{}] * length
        for i in range(length):
            dic = {}
            for key in cv.keys():
                dic[key] = cv.get(key)[i]
            output[i] = dic
        print(self.model_name, output)
        return output

    def generate_heatmap(self, label_predicted: list, label_test: list, categories: dict, figure_size=48,
                         output_path=None) -> Path:
        """
        Method which produces a specialized heatmap displaying the amount of predictions from the specified model and
        distinctly illustrates the number of correct predictions based on the shade of the color
        :param label_predicted: A list which consists of the elements that the model predicted overall
        :param label_test: A list that contains the elements which the model predicted based on the testing dataset
        :param categories: A dict which specifies the mapping of the integer id and their associated term
        :param figure_size: Specifies the height and width the generated graph
        :return: Path
        """
        prepare_results_directory()
        plt.figure(figsize=(figure_size, figure_size))
        annot_kws = {'fontsize': 20}
        mat = confusion_matrix(label_test, label_predicted)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
                    xticklabels=categories.values(),
                    yticklabels=categories.values(), annot_kws=annot_kws)

        plt.title(str(self.model_name) + " Predicted Values VS Expected Classifications", fontsize=48)
        plt.xlabel('Actual Classification', fontsize=32)
        plt.ylabel('Predicted Classification', fontsize=32)

        if output_path is None:
            output_path = Path(f'results/img/{self.model_name}_germanTrainingData_{str(self.dataset_type)}.png')

        plt.savefig(output_path)
        plt.cla()
        plt.clf()
        plt.close()  # When printing images from multiple algo in succession, closing the plot MIGHT have side affects
        return output_path

    def __get_categories(self):
        return generate_category_labels_from_csv(self.location_of_categories)

    def generate_classification_report(self, features_test, labels_test, categories):
        if features_test is None or labels_test is None or categories is None:
            raise ValueError()
        y_predicted = self.model_object.predict(features_test)
        report = classification_report(labels_test, y_predicted, target_names=categories.values())

        return self.classification_report_to_list(report)

    @staticmethod
    def classification_report_to_list(report: str):
        report = report.split('\n')

        columns = report[0] = report[0].strip().split()
        columns.insert(0, 'Classification')
        report.pop(0)
        for i in range(0, len(report)):
            # Each line of output needs to be organized to follow the pattern:
            # ['Sign Name', num1, num2, num3, num4]. All rows will follow this pattern.
            report[i] = report[i].split()
            report[i][0:-4] = [' '.join(report[i][0:-4])]

            if len(report[i]) == 4:
                report[i].insert(-3, '')
                report[i][0] = report[i][2]
                report[i][2] = ''

        report = [r for r in report if r != ['']]
        column_val_map = {}

        for i in range(0, len(report)):
            for c, column in enumerate(columns):
                column_val_map[column] = report[i][c]
            report[i] = column_val_map
            column_val_map = {}

        return report

    # stores an object as bit inputs
    def serialize(self):
        """
        Method which serializes the given object its specified to
        :return: None
        """
        location = self.__get_model_location()
        serialize(file_location=location, obj=self)

    def test(self, features_test, labels_test, features, labels):
        print(self.model_name.upper(), " Score: ",
              str(format(self.score(features_test, labels_test) * 100, ".5f")).strip(),
              "% Accuracy")  # add option to add params

        label_predicted = self.predict(features_test)  # todo: Is there a way to get this from predict_dataset?
        print(self.model_name.upper(), " Cross Validation: ",
              str(self.cross_validate(features, labels)))  # add option to add params
        print(self.generate_classification_report(features_test, labels_test, self.__get_categories()))

        cat = self.__get_categories()
        print(self.model_name.upper(), " Heatmap saved to: " + str(
            self.generate_heatmap(label_predicted, labels_test, cat)))

    def __prepare_locations(self, is_color):
        dataset_type = self.__set_colored_settings(is_color)
        location_of_categories = Path('datasets/SignNames.csv')
        location_of_dataset = Path(f'datasets/germanTrainingData_{str(dataset_type)}.pickle')

        return location_of_dataset, location_of_categories

    def __str__(self):
        return f'{self.model_name}_{self.dataset_type}'

    @staticmethod
    def __set_colored_settings(is_color):
        """
        :param is_color: bool identifying if the model is trained on a RGB or grayscale dataset.
        :return: str returns "colored"/"grayscale" used to identify and separate models trained for RGB or Grayscale.
        """
        return "colored" if is_color else "grayscale"

    @staticmethod
    def _timer(func, *args, **kwargs):
        start_time = time.perf_counter()  # Time at the start of execution
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # Time at the end of execution

        if type(value) is np.float64:
            value = float(value)

        if type(value) is not float:
            return value, end_time - start_time
        else:
            return float(format(float(value), '.6f')), float(format((end_time - start_time), '.6f'))
