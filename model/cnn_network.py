import numpy as np
import tensorflow as tf
from typing import Union


class CNNNetwork(tf.keras.Model):
    def __init__(
        self,
        grid_size,
        action_size,
        learning_rate,
        regularization_factor,
        *args,
        **kwargs
    ):
        super(CNNNetwork, self).__init__()
        # initialize model with 6 layers
        # input layer: (4, 4, 1) represent a (4,4) board and channel 1
        # layer1 CNN: 32 3x3 kernel, ReLU activation, stride 1, padding 1
        # layer2 CNN: 64 3x3 kernel, ReLU activation, stride 1, padding 1
        # Flatten Layer is used to turn output with shape (4, 4, 64) from layer2 into shape (1024, )
        # layer3 Full Connect: 512 neurons, ReLU activation
        # layer4 Full Connect: 128 neurons, ReLU activation
        # output layer: 4 neurons (action 0,1,2,3), linear activation
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    bias_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    activation="relu",
                    padding="same",
                    input_shape=(1, grid_size, grid_size),
                ),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    64,
                    (3, 3),
                    kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    bias_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    activation="relu",
                    padding="same",
                ),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    512,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    bias_regularizer=tf.keras.regularizers.l2(regularization_factor),
                ),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    bias_regularizer=tf.keras.regularizers.l2(regularization_factor),
                ),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    action_size,
                    activation="linear",
                    kernel_regularizer=tf.keras.regularizers.l2(regularization_factor),
                    bias_regularizer=tf.keras.regularizers.l2(regularization_factor),
                ),
            ]
        )

        self.loss = tf.keras.losses.mean_absolute_error
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        self.compile(optimizer=self.optimizer, loss=self.loss)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)
