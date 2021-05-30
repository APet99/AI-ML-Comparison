"""
cnn_pipeline.py

cnn model architecture

"""

from kerastuner import HyperModel
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class NeuralNetwork(HyperModel):

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """
        Builds a convolutional model

        :param hp: A HyperParameters instance
        :return: A model instance

        more information about model implementation
        https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
        https://keras-team.github.io/keras-tuner/

        """
        model = Sequential()

        model.add(Conv2D(filters=hp.Int('CONV_1_FILTER', min_value=32, max_value=64),
                         kernel_size=hp.Choice('KERNEL_1_FILTER', values=[3, 5]),
                         activation='relu',
                         input_shape=self.input_shape,
                         padding='same',
                         kernel_regularizer=l2(0.0005)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('DROPOUT_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Conv2D(filters=hp.Int('CONV_2_FILTER', min_value=32, max_value=128),
                         kernel_size=hp.Choice('KERNEL_2_FILTER', values=[3, 5]),
                         activation='relu',
                         padding='same',
                         kernel_regularizer=l2(0.0005)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('DROPOUT_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Conv2D(filters=hp.Int('CONV_3_FILTER', min_value=32, max_value=128),
                         kernel_size=hp.Choice('KERNEL_3_FILTER', values=[3, 5]),
                         activation='relu',
                         padding='same',
                         kernel_regularizer=l2(0.0005)))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float('DROPOUT_3', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Flatten())

        model.add(Dense(hp.Int('DENSE_1_LAYER', min_value=32, max_value=512), activation=hp.Choice(
            'dense_activation', values=['relu', 'tanh', 'sigmoid'],
            default='relu')))

        model.add(Dropout(hp.Float('DROPOUT_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Dense(self.num_classes, activation='softmax'))

        print("")
        # print("Compiling Model...")
        model.compile(Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model
