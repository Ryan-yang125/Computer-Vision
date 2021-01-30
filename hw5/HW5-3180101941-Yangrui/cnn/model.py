'''
Author: your name
Date: 2021-01-06 10:48:38
LastEditTime: 2021-01-07 21:48:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /cnn/model.py
'''
from tensorflow.keras import models, layers


def create_model(input_size=(32, 32, 1), kernel_size=(5, 5)):
    model = models.Sequential()

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    model.add(layers.Conv2D(filters=6, kernel_size=kernel_size,
                            activation='relu', input_shape=input_size))
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    model.add(layers.AveragePooling2D())

    # Layer 2: Convolutional. Output = 10x10x16.
    model.add(layers.Conv2D(
        filters=16, kernel_size=kernel_size, activation='relu'))
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    model.add(layers.AveragePooling2D())

    # Flatten. Input = 5x5x16. Output = 400.
    model.add(layers.Flatten())
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    model.add(layers.Dense(units=120, activation='relu'))
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    model.add(layers.Dense(units=84, activation='relu'))
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    model.add(layers.Dense(units=10, activation='softmax'))

    return model


def create_better_model():
    model = models.Sequential()

    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model
