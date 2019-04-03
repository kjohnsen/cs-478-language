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

rfe_keep = [False, False,  True, False,  True, False, False, False, False, False, False, False,
 False,  True, False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False,  True, False, False, False,
 False, False, False, False, False,  True, False, False, False, False,  True,  True,
 False, False, False, False, False, False, False, False, False, False, False, False,
 False,  True, False, False,  True,  True,  True, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False, False, False,
 False, False,  True,  True, False, False, False, False, False, False, False, False,
  True, False, False, False,  True, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False]
rfe_ranking = np.array([ 43,  48,   1,  20,   1,  38, 101,  99,  50,  86,  88,  75,  98,   1,  35,  19,  61,  65,
 102, 103,  80,   3,   7,  53,  58,  73,  45,  77, 104,  37,  30,  67,   1,  55,  33,  27,
  17,  26,  87,  60,  69,   1, 94,   2,  64,  36,   1,   1,   6,  79,  23,  44,  28,  93,
  66,  91,  32,  82,  85,  92,  96,   1,  10,   9,   1,   1,   1,  52,  22, 100,  74,  97,
  71,  16,  31,   8,  41,  14,  15,  57,  72,  70,  42,  78,  83,  95,   1,   1,  63,  89,
  29,  13,  62,  54,   5,  40,   1,  24,   4,  39,   1,  34,  51,  46,  47,  18,  11,  90,
  59,  76,  81,  25,  21,  49,  12,  56,  68,  84])

from nongram_features import data, labels

# filter for features
data = data[:, rfe_ranking < 90]

data /= np.max(np.abs(data), axis=0)
train_size = int(len(data) * 0.8)
train_data = data[0:train_size, :]
train_labels = labels[0:train_size]

test_data = data[train_size:, :]
test_labels = labels[train_size:]

model = keras.Sequential([
    keras.layers.Dense((np.size(data, 1) - 12)/2, input_shape=(len(train_data[0]), ), activation=tf.nn.relu),
    keras.layers.Dropout(.1),
    keras.layers.Dense(12, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_data, train_labels, validation_split=0.2, epochs=130)


test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
