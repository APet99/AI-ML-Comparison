"""
Dataset.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 2/13/2021
"""

import math
import os
import random
import statistics
from datetime import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.serialization import serialize


def feature_to_img(image_arr, color=False, img_size=32):
    img_full_range = image_arr * 255.0
    if color:
        img_arr_resized = img_full_range.reshape(img_size, img_size, 3).astype('uint8')
    else:
        img_arr_resized = img_full_range.reshape(img_size, img_size).astype('uint8')

    return Image.fromarray(img_arr_resized)


class Dataset:

    def __init__(self, is_color: bool, train_dir: Path = None, test_dir: Path = None, shuffle: bool = True,
                 dataset_name: str = "updated_germanTrainingDataset", sign_names=Path('datasets/SignNames.csv'),
                 test_csv=Path('datasets/Test.csv')):
        """
        Creates a dataset object containing features and labels for training and testing.
        :param is_color: bool determines if the dataset will be RGB, or converted to grayscale
        :param train_dir: path location of the training dataset
        :param test_dir: path location of the testing dataset
        :param shuffle: bool determines if the dataset should be shuffled upon initialization
        :param dataset_name: str name of the dataset (ex. germanTrainingDataset)
        """
        if train_dir is not None:
            self.is_color = is_color
            self.categories = self.generate_category_labels_from_csv(sign_names)
            self.classification_ideal_size = self.__calculate_balanced_dataset_size(train_dir=train_dir,
                                                                                    categories=self.categories)
            training_data = self.__create_training_data(train_dir)

            self.features, self.labels = self.__create_feature_label_arrays(training_data)
            self.features, self.labels = self.__flatten_training_arrays(self.features, self.labels)
            self.features = self.__reduce_dataset(self.features)
            self.test_csv_dir = test_csv
            if shuffle:
                self.__shuffle_training_data(training_data)

            if test_dir:
                test_data = self.__create_testing_data(test_dir=test_dir)
                self.features_test, self.labels_test = self.__create_feature_label_arrays(test_data)
                self.features_test, self.labels_test = self.__flatten_training_arrays(self.features_test,
                                                                                      self.labels_test)
                self.features_test = self.__reduce_dataset(self.features_test)

                self.features_train = self.features
                self.labels_train = self.labels
            else:
                self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
                    self.features, self.labels, train_size=0.80, stratify=self.labels)
            self.size_of_training = len(self.labels_train)
            self.size_of_testing = len(self.labels_test)

            self.creation_timestamp = dt.now()
            self.identifier = str(dataset_name + "_" + self.__set_colored_settings())
        else:
            raise ValueError('A train_dir is required to create a dataset.')

    def serialize(self, file_ending: str = ".pickle"):
        """
        Serializes the current Dataset to 'datasets\\{self.identifier}_{file_ending}'
        :return: None
        """
        serialize(file_location=os.path.join('datasets', (self.identifier + file_ending)), obj=self)

    def __create_training_data(self, train_dir, img_size=32):
        """Created A dictionary of training data, and the classification the data point belongs to.

        :param train_dir: The directory location of the training data.
        :type train_dir: str

        :param img_size: The pixel width and height each image should be resized to.
        :type img_size: int

        :return: Dictionary of features and labels of dataset.
        :rtype: dict
        """
        training_data = []
        for category in self.categories:
            path = os.path.join(train_dir, str(category))

            files = [f for f in os.listdir(path)]
            img_to_sample = np.random.choice(files, int(min(len(files), self.classification_ideal_size)))
            for img in img_to_sample:
                # print(img)
                try:
                    if self.is_color:
                        img = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        new_array = cv2.resize(rgb, (img_size, img_size))
                        # plt.imshow(new_array)
                        # plt.show()
                    else:
                        img_array = cv2.imread(os.path.join(path, img),
                                               cv2.IMREAD_GRAYSCALE)
                        new_array = cv2.resize(img_array, (img_size, img_size))
                        # plt.imshow(new_array, cmap="gray")
                        # plt.show()
                    training_data.append([new_array, category])
                except Exception as e:
                    # Do Noting
                    pass

        return training_data

    def __create_testing_data(self, test_dir, img_size=32):
        """Created A dictionary of testing data, and the classification the data points belongs to.

        :param test_dir: The directory location of the testing data.
        :type test_dir: Path

        :param img_size: The pixel width and height each image should be resized to.
        :type img_size: int

        :return: Dictionary of features and labels of dataset.
        :rtype: dict
        """

        testing_data = []

        link = self.test_csv_dir

        df = pd.read_csv(link, usecols=['ClassId', 'Path'])
        x = zip(df.ClassId, df.Path)
        tests_mapping = list(x)

        for img in tests_mapping:
            classification = img[0]

            path = os.path.join(test_dir, img[1][5:])
            try:
                if self.is_color:
                    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(rgb, (img_size, img_size))
                    # plt.imshow(new_array)
                    # plt.show()
                else:
                    img_array = cv2.imread(path,
                                           cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (img_size, img_size))
                    # plt.imshow(new_array, cmap="gray")
                    # plt.show()

                testing_data.append([new_array, classification])
            except Exception as e:
                # do nothing.
                pass
        return testing_data

    def __calculate_balanced_dataset_size(self, train_dir, categories):
        """Determines how many images should be used from each class of a dataset. If the number of images representing
        classifications are not balanced, the model would be unequally trained and present a bias when predicting.
        If a classification has marginally more or less data points than other classes,
        Then the resultant model would be skewed.

        :param train_dir: The directory of the training dataset.
        :type train_dir: str
        :param categories: A dictionary where the keys are class identifiers, and the values are class names.
        :type categories: dict
        :return: int The calculated number of images each classification should have to be represented equally.
        """
        data = self.__get_data_size_by_class(train_dir, categories)
        return math.ceil(statistics.mean([statistics.mode(data.values()), statistics.quantiles(data.values())[0]]))

    def __flatten_training_arrays(self, feature_array, labels_array):
        """Reduces the dimensions in the feature array. This is required for model's ability to interpret the data.
        Ex: 4-dimensional Array is reduced to a 3-dimensional array.

        :param feature_array: The multi-dimensional array containing an arrays of pixel values for each data point.

        :param labels_array: the mapping of features to the corresponding classification.

        :return: The features and labels arrays, with a lower dimension.

        """
        num_images = len(labels_array)
        if self.is_color:
            features = np.array(feature_array).reshape(num_images, -3)
        else:
            features = np.array(feature_array).reshape(num_images, -1)

        labels = np.array(labels_array).reshape(num_images, )
        return features, labels

    def __set_colored_settings(self):
        return "colored" if self.is_color else "grayscale"

    def __str__(self):

        return (f"\n* * * {self.identifier} * * *\n"
                f"The dataset is {self.__set_colored_settings().upper()}.\n"
                f"{self.size_of_training} Training images, with a max of {self.classification_ideal_size} images per classification.\n"
                f"{self.size_of_testing} Testing images.\n"
                f"Number of Classes: {len(self.categories)} \n"
                f"Dataset created at: {self.creation_timestamp}\n"
                f"\n")

    # Static Methods:
    @staticmethod
    def __create_feature_label_arrays(training_data):
        """Separates training data into features, and labels arrays.

        :param training_data: A list[list] of features and labels.
        :type training_data: list[list]

        :return: An array containing the features (data points) and the labels (what classification the data belongs to)
        """
        features = []
        labels = []

        for feature, label in training_data:
            features.append(feature)
            labels.append(label)
        return features, labels

    @staticmethod
    def generate_category_labels_from_csv(csv_dir):
        """Generates list of classification names from a CSV. Relates directory names with the english classification of
        classification is in the directory.

        Consider the following CSV file:
            ClassId     SignName
            0           Speed limit (20km/h)
            1           Speed limit (30km/h)
            2           Speed limit (50km/h)
            .
            .
            25          Road work
            .
            .

        :param csv_dir: The file location of the CSV file associating classID and classification name.
        :type csv_dir: str
        :return: Dictionary of class identifiers, with associated English classification. Where the keys are class
        identifiers (directory names), and the associated values are English class names.
        :rtype dictionary
        """
        if str(csv_dir)[-4:] != '.csv':
            raise ValueError('ERROR: A csv file must have a .csv file ending.')
        categories = {}
        df = pd.read_csv(csv_dir)

        for row in df.itertuples():
            categories[row[1]] = row[2]
        return categories

    @staticmethod
    def __shuffle_training_data(training_data):
        """Shuffles the training data from an organized state, to mixed order.

        :param training_data: An organized list of all training data built by classification.
        :type training_data: list
        :return: training_data with the contents in a shuffled order.
        :rtype: dict
        """
        random.shuffle(training_data)

    @staticmethod
    def __reduce_dataset(features):
        return features / 255.0

    @staticmethod
    def __get_data_size_by_class(train_dir, categories):
        """Counts how many data points each class has in the dataset.

        :param train_dir: The directory of the training dataset.
        :type train_dir: str

        :param categories: A dictionary where the keys are class identifiers, and the values are class names.
        :type categories: dict

        :return: A dictionary where keys are the class identifier, and the values are the number of data points in the
        dataset classification.
        :rtype: dict
        """
        size_count = {}
        for class_id in categories.keys():
            path = os.path.join(train_dir, str(class_id))

            num_files = next(os.walk(path))[2]
            size_count[class_id] = len(num_files)

        return size_count

    """
    Field Getters
    """

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def features_train(self):
        return self._features_train

    @property
    def labels_train(self):
        return self._labels_train

    @property
    def features_test(self):
        return self._features_test

    @property
    def labels_test(self):
        return self._labels_test

    @property
    def is_color(self):
        return self._is_color

    @property
    def size_of_training(self):
        return self._size_of_training

    @property
    def size_of_testing(self):
        return self._size_of_testing

    @property
    def identifier(self):
        return self._identifier

    """
    Field Setters
    """

    @features.setter
    def features(self, value):
        self._features = value

    @labels.setter
    def labels(self, value):
        self._labels = value

    @features_train.setter
    def features_train(self, value):
        self._features_train = value

    @labels_train.setter
    def labels_train(self, value):
        self._labels_train = value

    @features_test.setter
    def features_test(self, value):
        self._features_test = value

    @labels_test.setter
    def labels_test(self, value):
        self._labels_test = value

    @is_color.setter
    def is_color(self, value):
        self._is_color = value

    @size_of_training.setter
    def size_of_training(self, value):
        self._size_of_training = value

    @size_of_testing.setter
    def size_of_testing(self, value):
        self._size_of_testing = value

    @identifier.setter
    def identifier(self, value):
        self._identifier = value
