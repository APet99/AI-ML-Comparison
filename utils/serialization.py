#   Created By: Alex Peterson     AlexJoseph.Peterson@CalBaptist.edu
#   Last Edited: February 27, 2021 at 5:44 PM
#   Serialization of a Python object

import pickle

from utils.logger import log


def serialize(file_location, obj):
    """
    Preserves an object state by serializing and storing as a serialized local file.
    :return:
    """
    if obj is None:
        raise ValueError('ERROR: Can not serialize when object to serialize is None')
    with open(file_location, 'wb') as file:
        pickle.dump(obj, file=file)


def deserialize(file_location):
    """
    Converts a serialized file back into an object instance.
    :return:
    """
    try:
        with open(file_location, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        log("ERROR: The file attempting to be deserialized does not exist or can not be found.")
        return None
