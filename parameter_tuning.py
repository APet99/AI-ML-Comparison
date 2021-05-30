"""
runner.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 2/20/2021


Note: THIS IS A WORK IN PROGRESS. The following script likely wont work when ran.
Cause-- Absolute path and the file location of the used datasets.


Models to Implement:
    - Support Vector Machines (SVM)
    - K Nearest Neighbors
    - Random Forest
    - Multilayer Prerceptron
    - Decision Tree
"""
import os
import pathlib

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from utils.serialization import deserialize


def get_dataset_colorization(is_color: bool):
    return "colored" if is_color else "grayscale"


def prepare_results_directory():
    """
    Windows machines will throw errors if attempting to make a dir that already exists.
    """

    dirs = ['results', 'results/img', 'result/csv']
    for d in dirs:
        try:
            p = pathlib.Path(d)
            p.mkdir()
        except:
            print(d, 'Already exists.')


def main():
    is_colored_dataset = False
    prepare_results_directory()

    """
    A dataset can be initialized in one of two ways:

    - Clean, prepare, and initialize a data set from a local directory
    or
    - Deserialize an already prepared data set
    """
    dataset = deserialize(os.path.join('datasets', 'updated_germanTrainingDataset_' + get_dataset_colorization(
        is_colored_dataset) + '.pickle'))
    print(dataset.__str__())

    tuned_parameters = [{'C': [1, 10, 100, 1000], 'degree': [3, 25, 43], 'gamma': [0.001, 'scale', 'auto']}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(dataset.features_train, dataset.labels_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = dataset.labels_test, clf.predict(dataset.features_test)
        print(classification_report(y_true, y_pred))
        print()

    # print(knn.score(features_test=dataset.features_test, labels_test=dataset.labels_test))
    # print(rf.score(features_test=dataset.features_test, labels_test=dataset.labels_test))
    # print(mlp.score(features_test=dataset.features_test, labels_test=dataset.labels_test))
    # print(dec_tree.score(features_test=dataset.features_test, labels_test=dataset.labels_test))

    # If a model was trained, it can be saved for future use by serializing it.
    # svm.serialize()
    # knn.serialize()
    # rf.serialize()
    # mlp.serialize()
    # dec_tree.serialize()


if __name__ == '__main__':
    main()
