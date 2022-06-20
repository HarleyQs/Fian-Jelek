# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    # NORMALIZE YOUR IMAGE HERE
    x_train, x_test = x_train/255, x_test/255

    # DEFINE YOUR MODEL HERE
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # End with 10 Neuron Dense, activated by softmax

    # COMPILE MODEL HERE
    from tensorflow.keras.optimizers import RMSprop
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
