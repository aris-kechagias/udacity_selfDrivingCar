"""
https://keras.io/getting_started/faq/
https://keras-team.github.io/keras-tuner/

https://review.udacity.com/#!/rubrics/432/view
https://www.youtube.com/watch?v=rpxZ87YFg0M&feature=youtu.be&ab_channel=Udacity
Captured images: 160 x 320 x 3

Usage
--------------------------------------------------------
-Train the model in the workspace and save it.
-Download the save model to the local machine and run it (python drive.py model.h5 folder_name_of_saved_captures)
-The drive.py is running, waiting for the simulator to start in autonomous mode.The simulator in the same folder
-Create a video of driving in autonomous mode (python video.py folder_name_of_saved_captures). The video is named after the containing folder

In case of under-fitting: (low train and validation predictions)
    increase the number of epochs
    add more convolutions to the network.
In case of over-fitting: (high train but low validation predictions)
    use dropout or pooling layers
    use fewer convolution or fewer fully connected layers
    collect more data or further augment the data set
Note:
    If your model has low mean squared error on the training and validation sets but is driving off the track,
    this could be because of the data collection process.

cv2.imread will get images in BGR format, while drive.py uses RGB (inverse the channels)


Display training parameters
    history_object = model.fit_generator(..)
    print(history_object.history.keys())

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
"""

import csv
import cv2
import numpy as np
from math import ceil
from PIL import Image

import tensorflow._api.v2.compat.v1 as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2


import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


lines = []
server = '../../opt/myData/'
local = 'C:/playground/udacitySelfDrivingCar/imageProcessing/tensorFlow/BehavioralCloning/newData/'
with open(local + 'driving_log_all.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def process_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def generator(samples, batch=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch):
            batch_samples = samples[offset:offset+batch]

            imgs = []
            angles = []
            for batch_sample in batch_samples:
                location = local + 'IMG/'

                # read in images from center, left and right cameras
                # img_center = cv2.imread(location + batch_sample[0].split('\\')[-1])
                # img_left = cv2.imread(location + batch_sample[1].split('\\')[-1])
                # img_right = cv2.imread(location + batch_sample[2].split('\\')[-1])

                img_center = np.asarray(Image.open(location + batch_sample[0].split('\\')[-1]))
                img_left = np.asarray(Image.open(location + batch_sample[1].split('\\')[-1]))
                img_right = np.asarray(Image.open(location + batch_sample[2].split('\\')[-1]))

                # img_center = process_image(img_center)  # np.asarray( Image.open(path + row[0]))
                # img_left = process_image(img_left)
                # img_right = process_image(img_right)

                center_angle = float(batch_sample[3])
                correction = 0.17  # this is a parameter to tune # adjusted steering measurements for the side images
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # add images and angles to data set
                imgs.extend((img_center, img_left, img_right))
                angles.extend((center_angle, steering_left, steering_right))

                # imgs.append(img_center[..., ::-1])
                # angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(imgs)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


batch_size = 32  # effectively 96 as we use 3 views for each frame

train_samples, valid_samples = train_test_split(lines, test_size=0.3)

train_generator = generator(train_samples, batch=batch_size)
validation_generator = generator(valid_samples, batch=batch_size)

ch, row, col = 3, 70, 320  # Trimmed image format

# Objective: minimize the error between the steering measurements that the network predicts and the ground truth
model = Sequential()
model.add(Cropping2D(input_shape=(160, 320, 3), cropping=((63, 27), (0, 0))))  # top 70, bottom 25 pixels

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))  # or x / 255.0 - 0.5

model.add(Lambda(lambda image: tf.image.resize_images(image, (66, 200))))  # input for  = (66,200)

model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu'))

model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), input_shape=(31, 98, 24), activation='relu'))

model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), input_shape=(14, 47, 36), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(5, 22, 64), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(3, 20, 64), activation='relu'))

model.add(Flatten(input_shape=(1, 18, 64)))

model.add(Dense(1164))
model.add(Dropout(rate=0.2))

model.add(Dense(100))
model.add(Dropout(rate=0.2))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))
# Regression network predicting the steering measurement. No need for activation function, as in classification networks

# model.summary()

model.compile(loss='mse', optimizer='adam')  # mean sq. error instead of cross entropy because of the regression type

model.load_weights('./weights.05-0.03.hdf5')

# checkpoint = ModelCheckpoint(filepath='./weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True)  # if False every single epoch saves a model version
# stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)  # if accuracy is not decreased for delta for the last 'patience' epochs
#
# model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size),
#                     validation_data=validation_generator, validation_steps=ceil(len(valid_samples)/batch_size),
#                     epochs=5, verbose=1,  callbacks=[checkpoint, stopper])

model.save('model.h5')  # download on local machine and check how it drives the simulator
