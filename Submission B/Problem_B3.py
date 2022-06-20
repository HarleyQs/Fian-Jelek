# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator


def solution_B3():


    TRAINING_DIR = "data/rps/"
    training_datagen = ImageDataGenerator(rescale=1/255,
                                          shear_range=0.2,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          vertical_flip=True,
                                          zoom_range=0.2,
                                          rotation_range=40,
                                          fill_mode='nearest',
                                          validation_split=0.2
        )

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                           target_size=(150, 150),
                                                           subset='training',
                                                           batch_size=4,
                                                           class_mode='categorical')
    validation_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                                subset='validation',
                                                                batch_size=4,
                                                                target_size=(150, 150),
                                                                class_mode='categorical')


    model=tf.keras.models.Sequential([
    # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(62, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=120,
                        validation_steps=5,
                        epochs=20,
                        verbose=2)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
