import tensorflow._api.v2.compat.v1 as tf
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.preprocessing import LabelBinarizer

# Load pickled data
with open('./small-traffic-set/small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

# split data
X_train, y_train = data['features'], data['labels']

"""
Keras will automatically infer the shape of all layers after the first layer
sample Sequential() methods:
fit(), evaluate(), and compile()
"""
model = Sequential()  # wrapper for the neural network model

# 32 layers; filter_kernel (3,3)
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))  # before or ->after relu
model.add(Activation('relu'))

model.add(Flatten(input_shape=(32, 32, 3)))  # 1st Layer - Add a flatten layer

model.add(Dense(128))  # 2nd Layer - Add a fully connected layer
model.add(Activation('relu'))  # 3rd Layer - Add a ReLU activation layer

model.add(Dense(5))  # 4th Layer - Add a fully connected layer
model.add(Activation('softmax'))  # 5th Layer - Add a ReLU activation layer

# An Alternative Solution
# model = Sequential()
# model.add(Flatten(input_shape=(32, 32, 3)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(5, activation='softmax'))

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5)

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)


""" Testing """

with open('./small-traffic-set/small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5)
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("\nTesting")
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
