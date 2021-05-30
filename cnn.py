"""
cnn.py

trains and evaluate the cnn model

"""
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import ModelCheckpoint

import cnn_pipeline
from utils.preprocess_dataset import *

# HyperParameters
BATCH_SIZE = 100
EPOCHS = 50
INPUT_SHAPE = (32, 32, 1)
NUM_CLASSES = 43

# RandomSearch() method essentially searches for the best subset of hyperparameter values in some predefined model.
# this helps to improves accuracy of the model
# https://github.com/keras-team/keras-tuner/blob/master/kerastuner/tuners/randomsearch.py
# https://www.tensorflow.org/tutorials/keras/keras_tuner
# https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/

cnn_model = cnn_pipeline.NeuralNetwork(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
tuner = RandomSearch(cnn_model, objective='val_accuracy', max_trials=12, directory='./saved_models',
                     project_name="cnn_hypermodel")

cnn_model = cnn_pipeline.cnn_model
tuner = RandomSearch(cnn_model, objective='val_accuracy', max_trials=10, directory='./saved_models',
                     project_name="cnntest")

# tuner.results_summary()

# Performs a search for best hyperparameter configurations
tuner.search(x_train_final, y_train, epochs=12, validation_data=(x_val_final, y_valid))

# The models are loaded with the weights corresponding to their best
# checkpoint, at the end of the best epoch of best trial
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of neural network to view the overall architecture refer to the saved model folder. there are two images
# which demonstrate the layers of the model
best_model.summary()

'''create a path to save the model uncomment this block to rerun and train the model again'''
# https://www.tensorflow.org/guide/keras/save_and_serialize
filepath = "./saved_models/cnn.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callback = [model_checkpoint]

# Training begins by calling the fit() method
# explains more about steps_per_epoch
# https://datascience.stackexchange.com/questions/47405/what-to-set-in-steps-per-epoch-in-keras-fit-generator
# history = best_model.fit(image_augmentation.flow(x_train_final, y_train, batch_size=BATCH_SIZE),
#                          steps_per_epoch=int(np.ceil(len(x_train_final) // float(BATCH_SIZE))),
#                          epochs=EPOCHS,
#                          validation_data=(x_val_final, y_valid),
#                          shuffle=True,
#                          callbacks=callback)

# Plot the accuracies of the trained model
# train_accuracy = history.history['accuracy']
# validation_accuracy = history.history['val_accuracy']
# filepath = "./saved_models/cnn_model_test.hdf5"
# model_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callback = [model_checkpoint]
# #
# #
# # # hyperparameter these could probably be further tuned for better accuracies
# BATCH_SIZE = 100
# EPOCHS = 35
# #
# #
# # # Training begins by calling the fit() method
# history = final_model.fit(data_generator.flow(x_train_final, y_train, batch_size=BATCH_SIZE),
#                           steps_per_epoch=int(np.ceil(len(x_train_format) / float(BATCH_SIZE))),
#                           epochs=EPOCHS,
#                           validation_data=(x_val_final, y_valid),
#                           shuffle=True,
#                           callbacks=callback)
#
# #
# # # Plot the accuracies of the trained model
# train_accuracy_list = history.history['accuracy']
# validation_accuracy_list = history.history['val_accuracy']
# epochs_range = range(EPOCHS)
#
# plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
# plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
# plt.title('CNN Model Accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.show()


# Load the model's weights and biases for evaluation on test images
best_model.load_weights("./saved_models/cnn.hdf5")
best_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

prediction = best_model.evaluate(x_test_final, y_test, verbose=0)
print(" ")
print("Model Performance: ")
print("Test Accuracy ")
print("%s %.2f" % (best_model.metrics_names[1], prediction[1] * 100))
