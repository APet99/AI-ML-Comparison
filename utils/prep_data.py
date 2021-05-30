"""
prep_data.py

** Note: This file should no longer be nececarry. File is present util safe removal is guerenteed.

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 12/22/2020
"""
import os
import random
import statistics

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import log
from utils.serialization import serialize, deserialize


def generate_category_labels_from_csv(csv_dir):
    """Generates a list of classification names from a CSV. Relates directory names with the english classification of
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
    :return: A dictionary of class identifiers, with associated English classification names. Where the keys are class
    identifiers (directory names), and the associated values are English class names.
    :rtype dictionary
    """
    categories = {}
    df = pd.read_csv(csv_dir)

    for row in df.itertuples():
        categories[row[1]] = row[2]
    return categories


def remove_empty_entries_from_meta_csv(file_location):
    """Checks a CSV file, and if a row has an empty cell, the entire row is removed.

    :param file_location: The file location of the CSV to check.
    :type file_location: str
    :return None
    """

    data = pd.read_csv(file_location)
    data.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=None,
        inplace=True
    )
    log(data.shape)
    data.to_csv(file_location, index=False)


# Stats
def calculate_balanced_dataset_size(train_dir, categories):
    """Determines how many images should be used from each class of a dataset. If the number of images representing
    classifications are not balanced, the model would be unequally trained and present a bias when predicting.
    If a classification has marginally more or less data points than other classes, the resultant model would be skewed.

    :param train_dir: The directory of the training dataset.
    :type train_dir: str
    :param categories: A dictionary where the keys are class identifiers, and the values are class names.
    :type categories: dict
    :return: The calculated number of images each classification should have to be represented equally.
    """
    data = get_data_size_by_class(train_dir, categories)
    return statistics.mean([statistics.mode(data.values()), statistics.quantiles(data.values())[0]])


def get_data_size_by_class(train_dir, categories):
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


# noinspection TryExceptPass
def create_training_data(train_dir, categories, dataset_size, img_size, is_color):
    """Created A dictionary of training data, and the classification the data point belongs to.

    :param train_dir: The directory location of the training data.
    :type train_dir: str

    :param categories: A dictionary where the keys are class identifiers, and the values are class names.
    :type categories: dict

    :param dataset_size: The number of images each classification should have to be represented equally.
    :type dataset_size: int

    :param img_size: The pixel width and height each image should be resized to.
    :type img_size: int

    :param is_color: Indicates if a dataset should be in color, or converted to grayscale.
    :type is_color: bool

    :return: Dictionary of features and labels of dataset.
    :rtype: dict
    """
    training_data = []
    for category in categories:
        path = os.path.join(train_dir, str(category))

        itr = 1
        for img in os.listdir(path):
            if itr <= dataset_size:
                try:
                    if is_color:
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
                itr += 1
    return training_data


def shuffle_training_data(training_data):
    """Shuffles the training data from an organized state, to mixed order.

    :param training_data: An organized list of all training data built by classification.
    :type training_data: list
    :return: training_data with the contents in a shuffled order.
    :rtype: dict
    """
    random.shuffle(training_data)


def create_feature_label_arrays(training_data):
    """Separates training data into features, and labels arrays.

    :param training_data: A dictionary of features and labels.
    :type training_data: dict

    :return: An array containing the features (data points) and the labels (what classification the data belongs to)
    """
    features = []
    labels = []

    for feature, label in training_data:
        features.append(feature)
        labels.append(label)
    return features, labels


def flatten_training_arrays(feature_array, labels_array, is_color: bool):
    """Reduces the demensions in the feature array. This is required for model's ability to interpret the data.
    Ex: 4-demensional Array is reduced to a 3-demesional array.

    :param feature_array: The multi-demensional array containing an arrays of pixel values for each data point.

    :param labels_array: the mapping of features to the corresponding classification.

    :param is_color: Indicates if a dataset should be in color, or converted to grayscale.
    :type is_color: bool

    :return: The features and labels arrays, with a lower demension.

    """
    num_images = len(labels_array)
    if is_color:
        features = np.array(feature_array).reshape(num_images, -3)
    else:
        features = np.array(feature_array).reshape(num_images, -1)

    labels = np.array(labels_array).reshape(num_images, )
    return features, labels


def prepare_dataset_by_dir(train_dir, category_label_csv, is_color=True, pickle_out="", img_size=32, shuffle=True):
    """Prepares dataset from directory for use by machine learning models.

    :param train_dir: The directory of the training dataset.
    :type train_dir: str

    :param category_label_csv: The file location of the CSV file associating classID and classification name.
    :type category_label_csv: str

    :param is_color: Indicates if a dataset should be in color, or converted to grayscale.
    :type is_color: bool

    :param pickle_out: Optional Parameter: If specificed, the dataset will be serialized to the specified filepath.
    :type pickle_out: str

    :param img_size: Optional Parameter: The width and height each datapoint should be resized to.
    :type img_size: int

    :param shuffle: Optional Parameter: Indicates if the training data produced should be shuffled. shuffle=True by default.
    :type shuffle: bool

    :return: The prepared and ready to use feature array, and labels array.
    """
    categories = generate_category_labels_from_csv(category_label_csv)
    dataset_size = calculate_balanced_dataset_size(train_dir, categories)
    training_data = create_training_data(train_dir, categories, dataset_size, img_size, is_color)
    if shuffle:
        shuffle_training_data(training_data)
    if pickle_out:
        serialize(file_location=pickle_out, model_object=training_data)
    features, labels = create_feature_label_arrays(training_data)

    return flatten_training_arrays(features, labels, is_color)


def prepare_dataset_from_pickle(pickle_path: os.path, train_amount=0.80, is_color=True, shuffle=True):
    """Prepares a serialized dataset for use by machine learning models.

    :param cwd: The directory of the project on the local machine.
    :type cwd: str

    :param pickle_path: The folder and file name of the serialized object.
    :type pickle_path: str

    :param is_color: Optional Parameter: Indicates if a dataset should be in color, or converted to grayscale.
    :type is_color: bool


    :param shuffle: Optional Parameter: Indicates if the training data produced should be shuffled. shuffle=True by default.
    :type shuffle: bool

    :return: The prepared and ready to use feature array, and labels array.
    """
    training_data = deserialize(pickle_path)
    if shuffle:
        shuffle_training_data(training_data)
    features, labels = create_feature_label_arrays(training_data)

    features, labels = flatten_training_arrays(features, labels, is_color)

    features = features / 255.0

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                train_size=train_amount,
                                                                                stratify=labels)

    return features_train, features_test, labels_train, labels_test, features, labels


def reduce_dataset(data):
    return data / 255.0
