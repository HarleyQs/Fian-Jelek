# =====================================================================================
# PROBLEM A2 
#
# Build a Neural Network Model for Horse or Human Dataset.
# The test will expect it to classify binary classes. 
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy and validation_accuracy > 83%
# ======================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


def solution_A2():



    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR ='data/validation-horse-or-human'
    train_datagen = ImageDataGenerator(rescale=1/255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       vertical_flip=True,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest'
        )
    val_datagen = ImageDataGenerator(rescale=1 / 255)


    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        target_size=(150, 150),
                                                        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(VALIDATION_DIR,
                                                           target_size=(150, 150),
                                                           batch_size=4,
                                                           class_mode='binary')

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with a Neuron Dense, activated by sigmoid
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])


    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    history = model.fit(train_generator,
              steps_per_epoch =20,
              epochs =20,
              validation_data =validation_generator,
              validation_steps =5,
              verbose =2)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A2()
    model.save("model_A2.h5")
