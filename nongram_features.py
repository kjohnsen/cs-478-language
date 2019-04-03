from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

from sklearn.feature_selection import RFE
from sklearn.svm import SVR

data = pd.read_csv('features.csv')
le = LabelEncoder()
data['Language'] = le.fit_transform(data['Language'])

labels = data['Language'].to_numpy()
data = data.drop(['ID', 'Prompt', 'Language', 'Score Level'], axis=1)
print(data.head())

data = data.to_numpy()

if __name__ == '__main__':
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 15, step=1)
    selector = selector.fit(data, labels)
    print(selector.support_)
    print(selector.ranking_)


