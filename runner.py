"""
runner.py

Created By: Alex Peterson   AlexJoseph.Peterson@CalBaptist.edu
Last Edited: 2/27/2021


runner.py is used to test the performance of existing models.
The script logs the accuracy and time of model functionality.
The data generated is uploaded to Google Drive for further analysis.

The script runs all models (colored and grayscale) a specified number
of times. Runs consisting of multiple iterations is used to ensure
environment variables are consistent, removing the concern of comparing
model performance on various hardware configurations.

Before running runner.py, trained models are required. If the pickled versions of the classifier models are not in
'/models', they can be obtained by following the direction in build_train_models.py and generating the necessary files


The models being analysed:
    - Support Vector Machines (SVM)
    - K Nearest Neighbors
    - Random Forest
    - Multilayer Perceptron (MLP)
    - Decision Tree
"""

import os
import pathlib
import platform
import time
from subprocess import Popen

from utils.logger import log
from utils.results import CSVResult
from utils.results import ResultSaver as rS
from utils.serialization import deserialize


def _get_dataset_colorization(is_color: bool):
    return "colored" if is_color else "grayscale"


def prepare_results_directory():
    """
    Windows machines will throw errors if attempting to make a dir that already exists.
    """

    dirs = ['results/img/grayscale', 'results/img/colored', 'models']
    for d in dirs:
        p = pathlib.Path(d)
        p.mkdir(exist_ok=True, parents=True)


def main():
    # Starts by analyzing Grayscale models, and examines RGB images later.
    is_colored_dataset = False

    # Initialize the Grayscale Dataset
    dataset = deserialize(
        pathlib.Path(f'datasets/updated_germanTrainingDataset_{_get_dataset_colorization(is_colored_dataset)}.pickle'))

    # Load the grayscale models
    grayscale_svm = deserialize('models/svm_grayscale.pickle')
    grayscale_knn = deserialize('models/knn_grayscale.pickle')
    grayscale_rf = deserialize('models/rf_grayscale.pickle')
    grayscale_mlp = deserialize('models/mlp_grayscale.pickle')
    grayscale_dec_tree = deserialize('models/dt_grayscale.pickle')

    # Load the colored models
    colored_svm = deserialize('models/svm_colored.pickle')
    colored_knn = deserialize('models/knn_colored.pickle')
    colored_rf = deserialize('models/rf_colored.pickle')
    colored_mlp = deserialize('models/mlp_colored.pickle')
    colored_dec_tree = deserialize('models/dt_colored.pickle')

    # Identify and label models for iterations
    grayscale_models = [grayscale_svm, grayscale_knn, grayscale_rf, grayscale_mlp, grayscale_dec_tree]
    colored_models = [colored_svm, colored_knn, colored_rf, colored_mlp, colored_dec_tree]
    models = [grayscale_models, colored_models]
    # models = [[grayscale_knn], [colored_knn]]

    start = time.perf_counter()
    log(f'Execution Started: {start}')

    for model_set in models:

        log(f'{dataset}\n')
        for i in range(5):
            log(f'Starting Run: {i + 1}')
            score_results = []

            for model in model_set:
                log(f'{model} \t current runtime \t {time.perf_counter()}')

                # Predicts The Entire Dataset
                predict_results = model.predict_dataset(dataset.features_test, dataset.labels_test)
                log(f'{model} \t COMPLETED: Predict Dataset')

                # Generates Classification Report for each Sign
                classification_results = model.generate_classification_report(dataset.features_test,
                                                                              dataset.labels_test, dataset.categories)
                log(f'{model} \t COMPLETED: Classification Report')

                if i == 0 or i == 2 or i == 4:
                    # Cross validates the models against a 5 fold selection of the dataset
                    cross_validate_results = model.cross_validate(dataset.features, dataset.labels)
                    log(f'{model} \t COMPLETED: Cross Validate')

                # Generates an Average Accuracy for each model
                score = model.score(dataset.features_test, dataset.labels_test)
                log(f'{model} \t COMPLETED: Score')

                score_results.append(
                    {'model_name': model.model_name, 'average_accuracy': score[0], 'time_to_score': score[1],
                     'model_object': model.model_object})
                # Generate Heatmaps:
                heatmap_path = pathlib.Path(
                    f'results/img/{_get_dataset_colorization(is_colored_dataset)}/{model.model_name}_{i}')
                label_predicted, time_to_execute = model.predict(dataset.features_test)
                model.generate_heatmap(label_predicted, dataset.labels_test, dataset.categories,
                                       output_path=heatmap_path)
                log(f'{model} \t COMPLETED: Heatmap Generation')

                # Columns for each CSV report being generated
                predict_dataset_columns = predict_results[0].keys()
                classification_report_columns = classification_results[0].keys()
                cross_validate_columns = cross_validate_results[0].keys()
                score_columns = score_results[0].keys()

                # Storing the Prediction Results to CSV
                csv = CSVResult(
                    path=f'{_get_dataset_colorization(is_colored_dataset)}/{model.model_name}/prediction_results_{i}.csv',
                    column_names=predict_dataset_columns)
                for result in predict_results:
                    csv.add_result(result)
                rS.RESULTS_TO_SAVE.append(csv)

                # Storing the classification Results to CSV
                csv = CSVResult(
                    path=f'{_get_dataset_colorization(is_colored_dataset)}/{model.model_name}/classification_results_{i}.csv',
                    column_names=classification_report_columns)
                for result in classification_results:
                    csv.add_result(result)
                rS.RESULTS_TO_SAVE.append(csv)

                if i == 0 or i == 2 or i == 4:
                    # Storing the cross validation Results to CSV
                    csv = CSVResult(
                        path=f'{_get_dataset_colorization(is_colored_dataset)}/{model.model_name}/cross_validation_{i}.csv',
                        column_names=cross_validate_columns)
                    for result in cross_validate_results:
                        csv.add_result(result)
                    rS.RESULTS_TO_SAVE.append(csv)

            # Storing the Average Accuracy of each model to CSV
            score_csv = CSVResult(path=f'{_get_dataset_colorization(is_colored_dataset)}/score_results_{i}.csv',
                                  column_names=score_columns)
            for score in score_results:
                score_csv.add_result(score)
            rS.RESULTS_TO_SAVE.append(score_csv)

        # Do it all again, now in color :)
        is_colored_dataset = not is_colored_dataset
        dataset = deserialize(pathlib.Path(
            f'datasets/updated_germanTrainingDataset_{_get_dataset_colorization(is_colored_dataset)}.pickle'))
        log(f'{dataset}\n')

    log(f'* * * Generated all Reports * * *\nExecution took:{time.perf_counter() - start} seconds')


if __name__ == '__main__':
    '''Launches the Hardware Watchdog to monitor CPU & Memory Usage while running'''
    current_pid = os.getpid()
    kagrs = {}
    if platform.system() == 'Windows':
        from subprocess import DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW

        kagrs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)
    else:  # UNIX
        kagrs.update(start_new_session=True)

    tmp_pid = Popen(["python", "hardware_watchdog.py", "-p", str(os.getpid())], cwd=os.getcwd(), shell=True, **kagrs)
    ''''''

    rS.UPLOAD_RESULTS = True
    prepare_results_directory()
    log(f'Starting runner.py')
    main()
