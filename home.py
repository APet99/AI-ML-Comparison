from os import listdir, linesep
from os.path import join, isfile

import pandas as pd
import streamlit as st
from PIL import Image

st.title('ItsClassified')
selected_option = st.sidebar.selectbox('You\'re currently viewing...', ('Models', 'Datasets'))
st.sidebar.button("Go")

def model_chart_gen(model_to_gen):
    # loop through corresponding csv files for each of selected models and plot the "recall" values
    sizes = []
    datapoints = []
    updated_columns = []  # starting with index
    new_dict = {}
    for model in model_to_gen:
        if classification_data_paths.__contains__(model):
            df = pd.read_csv(filepath_or_buffer=classification_data_paths.get(model), usecols=csv_columns)
            # dataframes.append(df.rename(columns={'id' : 'index'}).set_index('index'))
            data = df['recall'].tolist()

            # clean up model csv metadata after line 42 in dataset and insert into dictionary
            if len(data) > 42:
                for i in range(42, len(data)):
                    data[i] = None
            filtered_data = list(filter(None, data))
            new_dict[model] = filtered_data

            datapoints.append(data)
            sizes.append(len(data))
            updated_columns.append(model)

    updated_dataframe = pd.DataFrame.from_dict(new_dict)
    st.title('Model Analysis Report')
    st.dataframe(updated_dataframe)
    st.line_chart(updated_dataframe)


if not selected_option == 'Datasets':
    st.title('Home')
    st.text('Welcome to the ItsClassified Website!')

    st.title('Implemented Machine Learning Models')
    st.info(
        "Here are the following Machine Learning Models we used in our implementation. To see a quick demonstration, click the 'Run' button.")

    st.sidebar.write('To see our blog posts, click [here](http://itsclassified-blog.ml/)')
    st.sidebar.write('To see our research paper, click [here](https://calbaptist-my.sharepoint.com/:w:/g/personal/660863_calbaptist_edu/EUlvrl8YG6ZBvrqCKYn_JsMBgGmwAzbqINEFepTzS6wz_A?e=O8QX2X)')

    # TODO: Implement "real-time" iterations on graph for corresponding models
    option = st.selectbox('Models to Read About',
                          ('Select a Model...', 'Support Vector Machines (SVM)', 'Random Forests (RF)',
                           'K-Nearest Neighbors (KNN)', 'Decision Tree (DT)',
                           'Multilayer Perceptron (MLP)'))
    if option == 'Support Vector Machines (SVM)':
        st.write('Support Vector Machines')
        st.write('The way that a Support Vector Machine works is given a point n, each data point is treated as if it '
                 'were a vector and place it on a graph. Then take a set w, this set is considered as the hyper plane of '
                 'the graph, then is multiplied by the set of x points, minus b, which is our offset. The points that '
                 'fall within this decision boundary are support vectors and are considered the same classified element. '
                 'Other data elements that are clumped together or within their own dataset boundary are considered data '
                 'of anther class or type.')
        svmImage = Image.open('model_images/svm.png')
        st.image(svmImage)

    if option == 'Random Forests (RF)':
        st.write('Random Forests')
        st.write('Random Forests work by using several different decision trees on various sub-examples of a given '
                 'dataset. Then by averaging our the results, the model\'s prediction is achieved.')
        rfImage = Image.open('model_images/randomForest.png')
        st.image(rfImage)

    if option == 'K-Nearest Neighbors (KNN)':
        st.write('K-Nearest Neighbors')
        st.write('K-Nearest Neighbors works by assigning objects to the class that most resembles its nearest '
                 'neighbors in the multi dimensional feature space. the number k is the number of neighboring objects '
                 'in the feature space, these neighbors are compared with the classified object. The actions that are '
                 'required to complete a prediction is: calculate the distance to each object in the training sample, '
                 'select k objects, and classify as the object with the least distance between the object in question '
                 'and the neighbor.')
        knnImage = Image.open('model_images/knnimage.PNG')
        st.image(knnImage)

    if option == 'Decision Tree (DT)':
        st.write('Decision Tree')
        st.write('Decision Trees are the easiest to visualize the decision making process. Decision Trees works by '
                 'predicting the class label of input, starting from the root of a tree. In order to use efficiently, '
                 'the possibilities are divided into smaller subsets based on a decision rule that is set for each '
                 'node. Every decision node has two or more branches and the leaves in a tree model contain decision '
                 'on what the outcome could possibly be.')
        dtImage = Image.open('model_images/descTree.PNG')
        st.image(dtImage)

    if option == 'Multilayer Perceptron (MLP)':
        st.write('Multilayer Perceptron')
        st.write('Multilayer Perceptron is a neural network model that maps a set of inputs to the corresponding '
                 'outputs. Specifically, MLP has multiple layers or nodes in a directed graph with each layer fully '
                 'connected to the next.')
        mlpImage = Image.open('model_images/mlpImage.png')
        st.image(mlpImage)

    if option == 'Select a Model...':
        st.error('Choose a model from the dropdown to read more about it')

    st.title("Generate Model Analysis")
    st.info("Select the models you would like to generate analysis reports for.")
    svm = st.checkbox('Support Vector Machines (SVM)')
    rf = st.checkbox('Random Forests (RF)')
    knn = st.checkbox('K-Nearest Neighbors (KNN)')
    dt = st.checkbox('Decision Tree (DT)')
    mlp = st.checkbox('Multilayer Perceptron (MLP)')

    classification_data_paths = {
        'rf': 'model_data/Fri Mar 12 015642 2021/rf_grayscale_classification_results_0.csv',
        'svm': 'model_data/Fri Mar 12 015642 2021/svm_grayscale_classification_results_0.csv',
        'knn': 'model_data/Fri Mar 12 015642 2021/knn_grayscale_classification_results_0.csv',
        'dt': 'model_data/Fri Mar 12 015642 2021/dt_grayscale_classification_results_0.csv',
        'mlp': 'model_data/Fri Mar 12 015642 2021/mlp_grayscale_classification_results_0.csv'
    }

    csv_columns = ['id', 'recall']

    score_models = []
    if st.button('Run Selected'):
        if svm:
            score_models.append('svm')
        if rf:
            score_models.append('rf')
        if knn:
            score_models.append('knn')
        if dt:
            score_models.append('dt')
        if mlp:
            score_models.append('mlp')

        if len(score_models) < 1:
            st.error("Please select at least one model to generate analysis report.")

    if st.button('Run All'):
        score_models = ['svm', 'rf', 'knn', 'dt', 'mlp']

    if len(score_models) > 0:
        model_chart_gen(score_models)

else:

    st.title('Sign Classes of The German Traffic Sign Benchmark Dataset')
    image_path = 'original_model_images/sign_images'
    sign_images = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))]
    st.write(linesep)
    st.write(linesep)
    st.image(sign_images)
