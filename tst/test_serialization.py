"""
test_serialization.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 3/07/2021
"""

import os
import shutil
import unittest
import warnings
from pathlib import Path

from utils import serialization

warnings.filterwarnings("ignore")


class TestClass:
    def __init__(self, name, age, available_schedule):
        self.name = name
        self.age = age
        self.available_schedule = available_schedule


class TestSerialization(unittest.TestCase):
    pickle_path = Path('tst_resources/myObject.pickle')

    def setUp(cls):
        if Path('tst_resources').is_dir():
            shutil.rmtree(Path('tst_resources'))
        Path('tst_resources').mkdir(exist_ok=False, parents=True)

    @classmethod
    def tearDownClass(cls):
        if str(Path('').absolute())[-3:] == 'tst':
            list_of_dir_to_remove = []
            for f in os.listdir(Path()):
                if os.path.isdir(f):
                    list_of_dir_to_remove.append(f)
            for f in list_of_dir_to_remove:
                shutil.rmtree(Path(f))

    @staticmethod
    def __createClass():
        name = "testName"
        age = 42
        available_schedule = {"Monday": True, "Tuesday": False, "Wednesday": False, "Thursday": True, "Friday": False}
        return TestClass(name=name, age=age, available_schedule=available_schedule)

    def __create_pickle(self):
        test_object = self.__createClass()
        serialization.serialize(file_location=self.pickle_path, obj=test_object)

    def test_serialize_no_location(self):
        test_object = self.__createClass()

        with self.assertRaises(Exception):
            serialization.serialize(None, test_object)

    def test_serialize_no_object(self):
        with self.assertRaises(ValueError):
            serialization.serialize("tst_resources/myObject.pickle", None)

    def test_serialize_str(self):
        test_object = self.__createClass()
        location = str(Path('tst_resources/myObject.pickle'))

        try:
            serialization.serialize(file_location=location, obj=test_object)
        except Exception:
            self.fail('ATTENTION: Serialization with a string location is not working!')

    def test_serialize_path(self):
        test_object = self.__createClass()
        location = Path('tst_resources/myObject.pickle')

        try:
            serialization.serialize(file_location=location, obj=test_object)
        except Exception:
            self.fail('ATTENTION: Serialization with a Path object is not working!')

    def test_serialize_overwrite(self):
        test_object = self.__createClass()
        test_object2 = self.__createClass()

        test_object2.name = "Stewart"
        test_object2.age = 23
        serialization.serialize(file_location=self.pickle_path, obj=test_object)

        try:
            serialization.serialize(file_location=self.pickle_path, obj=test_object2)
        except Exception:
            self.fail('ATTENTION: Serialization is not overwriting previous file!')

    def test_deserialize_no_location(self):
        self.__create_pickle()

        with self.assertRaises(Exception):
            test_obj = serialization.deserialize(file_location=None)

    def test_deserialize(self):
        self.__create_pickle()
        try:
            test_obj = serialization.deserialize(file_location=self.pickle_path)
        except Exception:
            self.fail()

    def test_deserialize_type(self):
        self.__create_pickle()

        test_obj = serialization.deserialize(file_location=self.pickle_path)
        self.assertTrue(isinstance(test_obj, TestClass))

    def test_deserialize_is_original(self):
        test_object = self.__createClass()
        serialization.serialize(file_location=self.pickle_path, obj=test_object)

        try:
            deserialized_object = serialization.deserialize(file_location=self.pickle_path)
            self.assertEqual(deserialized_object.name, test_object.name)
            self.assertEqual(deserialized_object.age, test_object.age)
            self.assertEqual(deserialized_object.available_schedule, test_object.available_schedule)

        except Exception:
            self.fail('ATTENTION: Deserialized object is different from the original object!')


if __name__ == '__main__':
    unittest.main()
