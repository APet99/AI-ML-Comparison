"""
test_dataset.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 3/07/2021
"""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np

from Dataset import Dataset
from utils.gdrive_utils.gdrive_utils import get_google_drive_file, get_repository_root_folder_path, \
    download_gdrive_folder


class TestDataset(unittest.TestCase):
    # TODO change to downloading with gdrive
    root_of_repo = get_repository_root_folder_path()
    raw_dataset_dir = root_of_repo.joinpath('raw dataset')
    if not raw_dataset_dir.exists():
        download_gdrive_folder(get_google_drive_file(file_name='raw dataset', is_file_a_folder=True), raw_dataset_dir)
    train_dir = raw_dataset_dir / 'archive' / 'Train'
    test_dir = raw_dataset_dir / 'archive' / 'Test'
    sign_names_path = Path('../datasets/SignNames.csv')
    test_csv_path = Path('../datasets/Test.csv')
    test_dataset = Dataset(is_color=False, train_dir=train_dir, sign_names=sign_names_path, test_csv=test_csv_path)

    """
    Test Resources
    """

    @staticmethod
    def create_dataset_location():
        dataset_path = Path('datasets')
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
            # root_path = get_repository_root_folder_path() + 'datsets'
            # files = [str(root_path) + 'SignNames.csv', str(root_path) + 'Test.csv']
            # for f in files:
            #     shutil.copy(f, dataset_path)

    def setUp(cls):
        if Path('tst_resources').is_dir():
            shutil.rmtree(Path('tst_resources'))
        Path('tst_resources').mkdir(exist_ok=False, parents=True)
        cls.create_dataset_location()

    @classmethod
    def tearDownClass(cls):
        if str(Path('').absolute())[-3:] == 'tst':
            list_of_dir_to_remove = []
            for f in os.listdir(Path()):
                if os.path.isdir(f):
                    list_of_dir_to_remove.append(f)
            for f in list_of_dir_to_remove:
                shutil.rmtree(Path(f))
            if Path('random_file.txt').exists():
                os.remove(Path('random_file.txt'))

    """
    constructor
    """

    def test_constructor_without_train_dir(self):
        with self.assertRaises(ValueError):
            test_dataset = Dataset(is_color=False, train_dir=None, test_dir=self.test_dir)

    def test_constructor_with_test_dir(self):

        try:
            test_dataset = Dataset(is_color=False, train_dir=self.train_dir, test_dir=self.test_dir,
                                   sign_names=self.sign_names_path, test_csv=self.test_csv_path)
        except Exception as e:
            self.fail()

    def test_constructor_without_test_dir(self):
        try:
            test_dataset = Dataset(is_color=False, train_dir=self.train_dir, test_dir=None,
                                   sign_names=self.sign_names_path)
        except Exception as e:
            self.fail()

    def test_constructor_grayscale(self):
        try:
            test_dataset = Dataset(is_color=False, train_dir=self.train_dir, test_dir=None,
                                   sign_names=self.sign_names_path)
            self.assertFalse(test_dataset.is_color)
        except Exception as e:
            self.fail()

    def test_constructor_colored(self):
        try:
            test_dataset = Dataset(is_color=True, train_dir=self.train_dir, test_dir=None,
                                   sign_names=self.sign_names_path)
            self.assertTrue(test_dataset.is_color)
        except Exception as e:
            self.fail()

    """
    serialize()
    """

    def test_serialize(self):
        test_dataset = Dataset(is_color=False, train_dir=self.train_dir, sign_names=self.sign_names_path,
                               test_csv=self.test_csv_path)

        try:
            # serialize
            test_dataset.serialize()
            self.assertTrue(Path('datasets/updated_germanTrainingDataset_grayscale.pickle').is_file())
        except Exception as e:
            self.fail()

    """
    __str__()
    """

    def test_to_string(self):
        test_dataset = Dataset(is_color=False, train_dir=self.train_dir, sign_names=self.sign_names_path,
                               test_csv=self.test_csv_path)
        expected_string = f"\n* * * {test_dataset.identifier} * * *\n" \
                          f"The dataset is GRAYSCALE.\n" \
                          f"{test_dataset.size_of_training} Training images, with a max of {test_dataset.classification_ideal_size} images per classification.\n" \
                          f"{test_dataset.size_of_testing} Testing images.\n" \
                          f"Number of Classes: {len(test_dataset.categories)} \n" \
                          f"Dataset created at: {test_dataset.creation_timestamp}\n" \
                          f"\n"

        self.assertEqual(test_dataset.__str__(), expected_string)

    """
    generate_category_label()
    """

    def test_generate_category_label_no_path(self):
        with self.assertRaises(ValueError):
            Dataset.generate_category_labels_from_csv(None)

    def test_generate_category_label_non_csv(self):
        path = Path('random_file.txt')
        file = open(path, 'a')
        file.close()

        with self.assertRaises(ValueError):
            Dataset.generate_category_labels_from_csv(path)

    def test_generate_category_label(self):
        path = self.sign_names_path

        try:
            Dataset.generate_category_labels_from_csv(path)
        except Exception:
            self.fail()

    """
    Getters:
    """

    def test_get_features(self):
        features = self.test_dataset.features

        self.assertIsNotNone(features)
        self.assertEqual(type(features), np.ndarray)

    def test_get_labels(self):
        labels = self.test_dataset.labels

        self.assertIsNotNone(labels)
        self.assertEqual(type(labels), np.ndarray)

    def test_get_features_train(self):
        features = self.test_dataset.features_train

        self.assertIsNotNone(features)
        self.assertEqual(type(features), np.ndarray)

    def test_get_labels_train(self):
        labels = self.test_dataset.labels_train

        self.assertIsNotNone(labels)
        self.assertEqual(type(labels), np.ndarray)

    def test_get_features_test(self):
        features = self.test_dataset.features_test

        self.assertIsNotNone(features)
        self.assertEqual(type(features), np.ndarray)

    def test_get_labels_test(self):
        labels = self.test_dataset.labels_test

        self.assertIsNotNone(labels)
        self.assertEqual(type(labels), np.ndarray)

    def test_get_is_color(self):
        is_color = self.test_dataset.is_color

        self.assertEqual(type(is_color), bool, f'is_color is type {type(is_color)} and expected type {bool}')
        self.assertFalse(is_color)

    def test_get_size_of_training(self):
        features_length = len(self.test_dataset.features_train)
        labels_length = len(self.test_dataset.labels_train)

        self.assertEqual(features_length, labels_length)
        self.assertEqual(type(features_length), int)
        self.assertEqual(type(labels_length), int)

    def test_get_size_of_testing(self):
        features_length = len(self.test_dataset.features_test)
        labels_length = len(self.test_dataset.labels_test)

        self.assertEqual(features_length, labels_length)
        self.assertEqual(type(features_length), int)
        self.assertEqual(type(labels_length), int)

    def test_get_identifier(self):
        identifier = self.test_dataset.identifier
        expected_identifier = 'updated_germanTrainingDataset_grayscale'

        self.assertEqual(identifier, expected_identifier,
                         f'The actual identifier is "{identifier}" but expected "{expected_identifier}".')

    """
    Setters:
    """

    def test_set_features(self):
        self.test_dataset.features = 72

        self.assertEqual(type(self.test_dataset.features), int)
        self.assertEqual(self.test_dataset.features, 72)

    def test_set_labels(self):
        self.test_dataset.labels = 72

        self.assertEqual(type(self.test_dataset.labels), int)
        self.assertEqual(self.test_dataset.labels, 72)

    def test_set_features_train(self):
        self.test_dataset.features_train = 72

        self.assertEqual(type(self.test_dataset.features_train), int)
        self.assertEqual(self.test_dataset.features_train, 72)

    def test_set_labels_train(self):
        self.test_dataset.labels_train = 72

        self.assertEqual(type(self.test_dataset.labels_train), int)
        self.assertEqual(self.test_dataset.labels_train, 72)

    def test_set_features_test(self):
        self.test_dataset.features_test = 72

        self.assertEqual(type(self.test_dataset.features_test), int)
        self.assertEqual(self.test_dataset.features_test, 72)

    def test_set_labels_test(self):
        self.test_dataset.labels_test = 72

        self.assertEqual(type(self.test_dataset.labels_test), int)
        self.assertEqual(self.test_dataset.labels_test, 72)

    def test_set_is_color(self):

        self.assertFalse(self.test_dataset.is_color)

        self.test_dataset.is_color = True
        self.assertTrue(self.test_dataset.is_color)

    def test_set_size_of_training(self):
        self.test_dataset.size_of_training = 1337

        self.assertEqual(type(self.test_dataset.size_of_training), int)
        self.assertEqual(self.test_dataset.size_of_training, 1337)

    def test_set_size_of_testing(self):
        self.test_dataset.size_of_testing = 1337

        self.assertEqual(type(self.test_dataset.size_of_testing), int)
        self.assertEqual(self.test_dataset.size_of_testing, 1337)

    def test_set_identifier(self):
        expected = 'newIdentifier'
        self.test_dataset.identifier = expected

        self.assertEqual(self.test_dataset.identifier, expected)


if __name__ == '__main__':
    unittest.main()
