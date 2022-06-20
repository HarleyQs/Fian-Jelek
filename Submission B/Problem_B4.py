# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pandas as pd


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE

    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE

    sentences = bbc['text']
    labels = bbc['category']
    # Using "shuffle=False"

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    label_word_index = label_tokenizer.word_index
    label_sequences = label_tokenizer.texts_to_sequences(labels)

    train_size = int(len(sentences) * training_portion)
    train_sentences = train_padded[:train_size]
    train_labels = label_sequences[:train_size]
    validation_sentences = train_padded[train_size:]
    validation_labels = label_sequences[train_size:]

    training_label_sequences = np.array(train_labels)
    validation_label_sequences = np.array(validation_labels)

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            if(logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):
                self.model.stop_training = True

    callbacks = myCallback()



    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.fit(train_sentences,
              training_label_sequences,
              epochs=75,
              validation_data=(validation_sentences, validation_label_sequences),
              verbose=1,
              batch_size=1,
              callbacks=callbacks)
    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.


if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
