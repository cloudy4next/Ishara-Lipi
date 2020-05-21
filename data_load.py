


import os
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
x = np.load('/home/cloudy/Desktop/bangla-lipi/x_char_bangla.npy', allow_pickle=True)
y = np.load('/home/cloudy/Desktop/bangla-lipi/y_char_bangla.npy', allow_pickle=True)

images = x
images = np.array([np.reshape(i, (64, 64)) for i in images])
images = np.array([i.flatten() for i in images])

label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify=labels, random_state = 101)

x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
