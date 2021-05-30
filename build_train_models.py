"""
build_train_models.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 2/28/2021

The following script builds, and trains 5 classification models against the German Signs Dataset

The following models are trained and serialized to the '/models' directory:
    Models Trained for RGB images:
        - Support Vector Machines (SVM)
        - K Nearest Neighbors
        - Random Forest
        - Multilayer Perceptron
        - Decision Tree
    Models Trained for Grayscale images:
        - Support Vector Machines (SVM)
        - K Nearest Neighbors
        - Random Forest
        - Multilayer Perceptron
        - Decision Tree

Steps to follow:
1) Run the following command to install all necessary dependencies:
    - On MAC or Linux run "pip3 install -r requirements.txt"

    - On Windows run "pip install -r requirements.txt"

2) Download the following files to the '/datasets' directory.
    * In a terminal, cd to the '/datasets' directory of the project.
    * Execute the following commands to download the colored and grayscale datasets:
        gdown https://drive.google.com/uc?id=1YwylpB1YTGxCg7Oq0oZyEWg7HSMYkvzA
        gdown https://drive.google.com/uc?id=1lfwlpNIzsRjSgacvWy00jDWilAk6T2y7

    the contents of '/datasets' should be the following:
    - SignNames.csv
    - Test.csv
    - updated_germanTrainingDataset_colored.pickle
    - updated_germanTrainingDataset_grayscale.pickle

3) Run this script though an IDE or a command line.

After the script finishes, the '/models' directory of the project should contain all of the serialized models generated.
"""

import pathlib

from sklearn.ensemble import RandomForestClassifier as RandFor
from sklearn.neighbors import KNeighborsClassifier as KNeighbor
from sklearn.neural_network import MLPClassifier as Mlp
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DecTree

from Model import Model
from utils.gdrive_utils.gdrive_utils import get_repository_root_folder_path, download_gdrive_folder, \
    get_google_drive_file
from utils.logger import log
from utils.serialization import deserialize


def _get_dataset_colorization(is_color: bool):
    return "colored" if is_color else "grayscale"


def prepare_models_directory():
    p = pathlib.Path('models')
    p.mkdir(exist_ok=True, parents=True)


def download_datasets_folder():
    log(f'Could not find datasets folder. \t Downloading folder from Google Drive.')
    download_gdrive_folder(get_google_drive_file(file_name='datasets', is_file_a_folder=True), dataset_folder_path)
    log(f'Done downloading from Google Drive.')


def main():
    is_colored_dataset = False

    for i in range(2):
        log('Beginning model training')

        dataset = deserialize(pathlib.Path(
            f'datasets/updated_germanTrainingDataset_{_get_dataset_colorization(is_colored_dataset)}.pickle'))
        """
        The optimal model parameters found are listed below:

        # svm Optimal: C=1000, kernel='rbf', gamma=0.001
        # KNN optimal:'algorithm': 'auto', 'n_jobs': -1, 'n_neighbors': 1
        # Random Forest Optimal: 'criterion': 'entropy', 'n_estimators': 1000, 'n_jobs': -1
        # MLP optimal: 'activation': 'logistic', 'learning_rate': 'constant', 'max_iter': 10000, 'solver': 'adam'
        # DT optimal: 'criterion': 'entropy', 'max_features': None
        
        """

        # Create a new untrained model
        svm = Model("svm", model_object=SVC(C=1000, kernel='rbf', gamma=0.001), is_color=is_colored_dataset)
        knn = Model("knn", KNeighbor(algorithm='auto', n_jobs=-1, n_neighbors=1), is_color=is_colored_dataset)
        rf = Model("rf", RandFor(n_estimators=75, n_jobs=-1, bootstrap=False, min_impurity_decrease=0.001,
                                 min_weight_fraction_leaf=0, criterion='entropy', max_features=500),
                   is_color=is_colored_dataset)
        mlp = Model("mlp", Mlp(activation='logistic', learning_rate='constant', max_iter=10000, solver='adam'),
                    is_color=is_colored_dataset)
        dec_tree = Model('dt', DecTree(criterion='entropy', max_features=None), is_color=is_colored_dataset)

        # Train the created models
        svm.fit(features_train=dataset.features_train, labels_train=dataset.labels_train)
        log(f' COMPLETED: Training of Model \t SVM.')

        knn.fit(features_train=dataset.features_train, labels_train=dataset.labels_train)
        log(f' COMPLETED: Training of Model \t KNN.')

        rf.fit(features_train=dataset.features_train, labels_train=dataset.labels_train)
        log(f' COMPLETED: Training of Model \t RF.')

        mlp.fit(features_train=dataset.features_train, labels_train=dataset.labels_train)
        log(f' COMPLETED: Training of Model \t MLP.')

        dec_tree.fit(features_train=dataset.features_train, labels_train=dataset.labels_train)
        log(f' COMPLETED: Training of Model \t DT.')

        # Store the trained models for future use
        svm.serialize()
        knn.serialize()
        rf.serialize()
        mlp.serialize()
        dec_tree.serialize()

        # Now we do it all again for colored images :)
        is_colored_dataset = not is_colored_dataset
    # print(f'All models were successfully trained, and serialized to {pathlib.Path("/models")}')
    log(f'All models were successfully trained, and serialized to {pathlib.Path("/models")}')


if __name__ == '__main__':
    prepare_models_directory()

    dataset_folder_path = get_repository_root_folder_path().joinpath('datasets')
    if not dataset_folder_path.exists():
        download_datasets_folder()

    main()
