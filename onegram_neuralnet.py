from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
# import matplotlib.pyplot as plt

print(tf.__version__)

onegram_data = pd.read_csv('reduced-1-grams.csv')
le = LabelEncoder()
onegram_data.iloc[:, -1:] = le.fit_transform(onegram_data.iloc[:, -1:])
# print(data.head())

labels = onegram_data.iloc[:, -1:].to_numpy()
onegram_data = onegram_data.iloc[:, 0:-1].to_numpy()
print(onegram_data[0:5, :])

from nongram_features import data as nongram_data

print(nongram_data.shape)
print(onegram_data.shape)
data = onegram_data
# data = np.concatenate(nongram_data, onegram_data, axis=1)

train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
train_labels = labels[0:train_size]

test_data = data[train_size:, :]
test_labels = labels[train_size:]

model = keras.Sequential([
    keras.layers.Dropout(.1),
    keras.layers.Dense((np.size(data, 1) + 12)/2, input_shape=(len(train_data[0]), ), activation=tf.nn.relu),
    keras.layers.Dropout(.1),
    # keras.layers.Dense((np.size(data, 1) + 12)*2/3, input_shape=(len(train_data[0]), ), activation=tf.nn.relu),
    # keras.layers.Dropout(.2),
    # keras.layers.Dense((np.size(data, 1) + 12)/3, input_shape=(len(train_data[0]), ), activation=tf.nn.relu),
    # keras.layers.Dropout(.2),
    keras.layers.Dense(12, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_data, train_labels, validation_split=0.2, epochs=20, callbacks=[keras.callbacks.EarlyStopping(patience=3)])


test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
