"""
test_model.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 2/27/2021
"""
import os
import shutil
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RandFor

from Dataset import Dataset
from Model import Model
from utils.gdrive_utils.gdrive_utils import get_repository_root_folder_path, get_google_drive_file, \
    download_gdrive_folder
from utils.serialization import deserialize

warnings.filterwarnings("ignore")


class TestModels(unittest.TestCase):
    """
    Test Resources
    """
    dataset_path = Path('tst_resources/test_dataset.pickle')
    model_path = Path('tst_resources/test_model.pickle')
    model = Model("rf",
                  RandFor(n_estimators=75, bootstrap=False, min_impurity_decrease=0.001, min_weight_fraction_leaf=0,
                          criterion='entropy', max_features=500), is_color=False)

    root_of_repo = get_repository_root_folder_path()
    raw_dataset_dir = root_of_repo.joinpath('raw dataset')
    if not raw_dataset_dir.exists():
        download_gdrive_folder(get_google_drive_file(file_name='raw dataset', is_file_a_folder=True), raw_dataset_dir)
    train_dir = raw_dataset_dir / 'archive' / 'Train'
    test_dir = raw_dataset_dir / 'archive' / 'Test'
    sign_names_path = Path('../datasets/SignNames.csv')
    test_csv_path = Path('../datasets/Test.csv')
    dataset = Dataset(is_color=False, train_dir=train_dir, sign_names=sign_names_path, test_csv=test_csv_path)

    @staticmethod
    def create_dataset_location():
        dataset_path = Path('datasets')
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setUp(cls):
        if Path('tst_resources').is_dir():
            shutil.rmtree(Path('tst_resources'))
        Path('tst_resources').mkdir(exist_ok=False, parents=True)
        cls.create_dataset_location()
        cls.model = deserialize(Path('../models/rf_grayscale.pickle'))

    @classmethod
    def tearDownClass(cls):
        if str(Path('').absolute())[-3:] == 'tst':
            list_of_dir_to_remove = []
            for f in os.listdir(Path()):
                if os.path.isdir(f):
                    list_of_dir_to_remove.append(f)
            for f in list_of_dir_to_remove:
                shutil.rmtree(Path(f))

    """
    constructor
    """

    def test_constructor_grayscale(self):
        try:
            test_model = Model("rf", RandFor(n_estimators=75, bootstrap=False, min_impurity_decrease=0.001,
                                             min_weight_fraction_leaf=0, criterion='entropy', max_features=500),
                               is_color=False)
            self.assertEqual(test_model.dataset_type, 'grayscale')
        except Exception:
            self.fail()

    def test_constructor_colored(self):
        try:
            test_model = Model("rf", RandFor(n_estimators=75, bootstrap=False, min_impurity_decrease=0.001,
                                             min_weight_fraction_leaf=0, criterion='entropy', max_features=500),
                               is_color=True)
            self.assertEqual(test_model.dataset_type, 'colored')
        except Exception:
            self.fail()

    def test_constructor_no_model_object(self):
        with self.assertRaises(ValueError):
            test_model = Model("rf", model_object=None,
                               is_color=False)

    def test_constructor_no_name(self):
        with self.assertRaises(TypeError):
            test_model = Model(RandFor(n_estimators=75, bootstrap=False, min_impurity_decrease=0.001,
                                       min_weight_fraction_leaf=0, criterion='entropy', max_features=500),
                               is_color=False)

    def test_constructor(self):
        test_model = Model("rf", RandFor(n_estimators=75, bootstrap=False, min_impurity_decrease=0.001,
                                         min_weight_fraction_leaf=0, criterion='entropy', max_features=500),
                           is_color=False)
        self.assertIsNotNone(test_model)
        self.assertTrue(isinstance(test_model.model_object, RandFor))
        self.assertEqual(test_model.model_name, 'rf')
        self.assertEqual(test_model.dataset_type, 'grayscale')

    """
    predict()
    """

    def test_predict_no_feature(self):
        with self.assertRaises(ValueError):
            self.model.predict(None)

    def test_predict_single(self):
        feature = self.dataset.features_test[0]
        feature = feature.reshape(1, -1)
        try:
            self.model.predict(feature=feature)
        except Exception:
            self.fail()

    def test_predict_all(self):
        features = self.dataset.features_test
        for f in features:
            feature = f.reshape(1, -1)

            try:
                self.model.predict(feature=feature)
            except Exception:
                self.fail()

    def test_predict_return(self):
        feature = self.dataset.features_test[0]
        feature = feature.reshape(1, -1)
        predicted_class, time = self.model.predict(feature=feature)

        self.assertEqual(np.int32, type(predicted_class[0]))
        self.assertEqual(float, type(time))

    """
    predict_dataset()
    """

    def test_predict_dataset_no_features(self):
        with self.assertRaises(TypeError):
            self.model.predict_dataset(None, self.dataset.labels_test)

    def test_predict_dataset_no_labels(self):
        with self.assertRaises(TypeError):
            self.model.predict_dataset(self.dataset.features_test, None)

    def test_predict_dataset_results(self):
        values = self.model.predict_dataset(self.dataset.features_test, self.dataset.labels_test)

        self.assertIsNotNone(values)
        self.assertEqual(list, type(values))

    """
    score()
    """

    def test_score_no_features(self):
        with self.assertRaises(TypeError):
            self.model.score(labels_test=self.dataset.labels_test)

    def test_score_no_labels(self):
        with self.assertRaises(TypeError):
            self.model.score(features_test=self.dataset.features_test)

    def test_score_results(self):
        accuracy, time = self.model.score(features_test=self.dataset.features_test,
                                          labels_test=self.dataset.labels_test)

        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(time)

        self.assertEqual(float, type(accuracy))
        self.assertEqual(float, type(time))

    """
    cross_validate()
    """

    def test_cross_validate_no_features(self):
        with self.assertRaises(TypeError):
            self.model.cross_validate(labels=self.dataset.labels)

    def test_cross_validate_no_labels(self):
        with self.assertRaises(TypeError):
            self.model.cross_validate(features=self.dataset.features)

    # def test_cross_validate_results(self):
    #     result = self.model.cross_validate(features=self.dataset.features, labels=self.dataset.labels)
    #     cross_validate_datapoints = ['fit_time', 'score_time', 'test_score']
    #
    #     self.assertIsNotNone(result)
    #     self.assertEqual(cross_validate_datapoints, list(result[0].keys()))
    #
    #     for point in cross_validate_datapoints:
    #         self.assertEqual(float, type(result[0].get(point)))
    #     self.assertEqual(list, type(result))

    """
    generate_heatmap()
    """

    def test_generate_heatmap_no_label_predicted(self):
        label_predicted = self.model.predict(self.dataset.features_test)

        with self.assertRaises(ValueError):
            self.model.generate_heatmap(label_predicted, self.dataset.labels_test, self.dataset.categories)

    def test_generate_heatmap_no_label_test(self):
        with self.assertRaises(ValueError):
            self.model.generate_heatmap(label_predicted=None, label_test=self.dataset.labels_test,
                                        categories=self.dataset.categories)

    def test_generate_heatmap_no_categories(self):
        label_predicted = self.model.predict(self.dataset.features_test)

        with self.assertRaises(ValueError):
            self.model.generate_heatmap(label_predicted=label_predicted, label_test=self.dataset.labels_test,
                                        categories=None)

    # def test_generate_heatmap_results(self):
    #     label_predicted = self.model.predict(self.dataset.features_test)
    #
    #     result = self.model.generate_heatmap(label_predicted=label_predicted, label_test=self.dataset.labels_test,
    #                                          categories=self.dataset.categories)
    #     self.assertEqual(type(Path), type(result))

    """
    generate_classification_report()
    """

    def test_generate_classification_report_no_features(self):
        with self.assertRaises(ValueError):
            self.model.generate_classification_report(features_test=None, labels_test=self.dataset.labels_test,
                                                      categories=self.dataset.categories)

    def test_generate_classification_report_no_labels(self):
        with self.assertRaises(ValueError):
            self.model.generate_classification_report(features_test=self.dataset.features_test,
                                                      labels_test=None,
                                                      categories=self.dataset.categories)

    def test_generate_classification_report_no_categories(self):
        with self.assertRaises(ValueError):
            self.model.generate_classification_report(features_test=self.dataset.features_test,
                                                      labels_test=self.dataset.labels_test,
                                                      categories=None)

    def test_generate_classification_report_results(self):
        result = self.model.generate_classification_report(features_test=self.dataset.features_test,
                                                           labels_test=self.dataset.labels_test,
                                                           categories=self.dataset.categories)

        self.assertEqual(type([dict]), type(result))

    """
    serialize()
    """

    def test_serialize(self):
        try:
            self.model.serialize()
            self.assertTrue(Path('models/rf_grayscale.pickle').is_file())
        except Exception as e:
            self.fail()

    """
    __str__()
    """

    def test_to_string(self):
        self.assertEqual('rf_grayscale', self.model.__str__())


if __name__ == '__main__':
    unittest.main()
