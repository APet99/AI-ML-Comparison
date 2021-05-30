"""
data prep and preprocessing steps used to train the cnn model
"""

import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from utils.prep_data import prepare_dataset_from_pickle, generate_category_labels_from_csv


def load_pickled_dataset(file_name, data_columns):
    """
    loads the pickled file and return the Unpickle dataset
    This is a pickled version of the (GTSRB) dataset in which the images have already has been resized to 32x32.
    The pickled dataset is a dictionary with 4 key/value pairs:
    :param file_name: pickled dataset to be opened
    :param data_columns: dictionary with 4 key / value pairs
    """
    with open(file_name, mode='rb') as pickle_file:
        dataset = pickle.load(pickle_file)
        return tuple(map(lambda x: dataset[x], data_columns))


DATASET_PATH = Path.cwd() / "datasets"

'''
this is the formatted training and testing dataset that is used for the the svm, k_neighbors, and naieve_bayes. 
in order to have a proper comparison, this dataset was also used to train and test the cnn model. 
'''
features, labels = prepare_dataset_from_pickle(os.getcwd(),
                                               os.path.join(DATASET_PATH, "./germanTrainingData_grayscale.pickle"),
                                               is_color=False)

features = features / 255.0
x_train_format, x_test_format, y_train_format, y_test_format = train_test_split(features, labels, train_size=0.80,
                                                                                stratify=labels)
categories = generate_category_labels_from_csv(os.path.join(DATASET_PATH, "./SignNames.csv"))

# to train the model with the non-formatted dataset un-comment this and use it. i used the same validation dataset for
# for both cases
# training_file = os.path.join(DATASET_PATH, "./train.p")
# testing_file = os.path.join(DATASET_PATH, "./test.p")
# validation_file = os.path.join(DATASET_PATH, "./valid.p")

# # un-comment this to extract the labels from the dictionary and use for non-formatted dataset
# x_train, y_train = load_pickled_dataset(training_file, ['features', 'labels'])
# x_test, y_test = load_pickled_dataset(testing_file, ['features', 'labels'])
# x_valid, y_valid = load_pickled_dataset(validation_file, ['features', 'labels'])

# preprocessed_pickle = os.path.join(DATASET_PATH, "./preprocessed_valid_data.pickle")
# if not os.path.isfile(preprocessed_pickle):
#     print("Saving data to pickle...")
#     try:
#         with open(preprocessed_pickle, 'wb') as pfile:
#             pickle.dump(
#                 {
#                     'valid_features': x_valid,
#                     'valid_labels': y_valid,
#                 }, pfile, protocol=2)
#     except Exception as e:
#         print("error unable to save pickle dataset", preprocessed_pickle, ':', e)
#         raise
# print("Data saved as pickle")


number_training = len(x_train_format)
# number_validation = len(x_valid)
number_testing = len(x_test_format)
number_classes = len(categories)


def dataset_summary():
    # ==== display a summary of the datasets ===
    print("Number of training examples =", number_training)
    # print("Number of validation examples =", number_validation)
    print("Number of testing examples =", number_testing)
    print("Number of classes =", number_classes)
    print("Training Pixel Values = min %.3f, max = %.3f" % (x_train_format.min(), x_train_format.max()))


dataset_summary()


def plot_histogram(dataset, label):
    unique_train, counts_train = np.unique(dataset, return_counts=True)
    plt.bar(unique_train, counts_train)
    plt.grid()
    plt.title("Distribution of " + label)
    plt.ylabel("Number of Images")
    plt.show()


# Plot histogram of datasets
# plot_histogram(y_train_format, "Training Dataset")
# plot_histogram(y_valid, "Validation Dataset")


def preprocess_images(image):
    """
    changes the range of pixel intensity values from (0, 255) to (0, 1)
    and convert colored images into grayscale, also enhances an image with low contrast, using a method called local
    histogram equalization, which spreads out the most frequent intensity values in an image improving the contrast
    :param image: 4D array containing raw pixel data of the traffic sign images
    :return: the preprocessing image format
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255.0
    return image


# x_train_final = np.array(list(map(preprocessing, x_train)))
# x_val_final = np.array(list(map(preprocess_images, x_valid)))
# x_test_final = np.array(list(map(preprocessing, x_test)))


# reshapes the rgb channel of the images to 1 after applying grayscale
x_train_final = x_train_format.reshape(9456, 32, 32, 1)
# x_val_final = x_val_final.reshape(4410, 32, 32, 1)
x_test_final = x_test_format.reshape(2364, 32, 32, 1)

# performs additional image processing and image augmentation such as rotating, scaling, and flipping images
# more information about this process
# https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
image_augmentation = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, shear_range=0.1,
                                        rotation_range=10,
                                        featurewise_center=False,
                                        featurewise_std_normalization=False)

image_augmentation.fit(x_train_final)


def plot_images_after_augmentation():
    aug_images = image_augmentation.flow(x_train_final, y_train_format, batch_size=3, save_to_dir=
    '../sample_images', save_prefix='img', save_format='jpg')

    for x_batch, y_batch in aug_images:
        for i in range(0, 3):
            plt.subplot(350 + 1 + i)
            plt.imshow(x_batch[i].reshape(32, 32), cmap=plt.get_cmap('gray'))
        plt.show()
        break


# plot_images_after_augmentation()

# apply one Hot Encode to the data labels. converts a numpy array into has binary values
# Here's an article that explains what is
# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
y_train = to_categorical(y_train_format, 43)
# y_valid = to_categorical(y_valid, 43)
y_test = to_categorical(y_test_format, 43)
