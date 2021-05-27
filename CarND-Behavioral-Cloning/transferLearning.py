"""
Generalized overview of convolutional network processing:
---------------------------------------------------------
first layer -> detect edges in the image
second layer -> detect shapes
third convolutional layer -> higher level features

Four Main Cases When Using Transfer Learning
---------------------------------------------------------
2 first cases: Because the data set is small, overfitting is still a concern.
To combat overfitting, the weights of the original neural network will be held constant, like in the first case.
But the original training set and the new data set do not share higher level features.
In this case, the new network will only use the layers containing lower level features.

new data set small AND similar to original training data
 ->slice off the end of the neural network
 ->add a new fully connected layer that matches the number of classes in the new data set
 ->randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
 ->train the network to update the weights of the new fully connected layer

new data set small AND different from original training data
 ->slice off most of the pre-trained layers near the beginning of the network
 ->add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new set
 ->randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
 ->train the network to update the weights of the new fully connected layer

new data set large AND similar to original training data
 ->remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
 ->randomly initialize the weights in the new fully connected layer
 ->initialize the rest of the weights using the pre-trained weights
 ->re-train the entire neural network

new data set large AND different from original training data
 ->remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
 ->retrain the network from scratch with randomly initialized weights
 ->alternatively, you could just use the same strategy as the "large and similar" data case
"""
import numpy as np

import tensorflow._api.v2.compat.v1 as tf

from keras.layers import Input, Lambda, Dense, GlobalAveragePooling2D

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model  # Imports the Model API https://keras.io/api/models/model/#fit_generator
from keras.preprocessing.image import ImageDataGenerator  # generator to pre-process our images for ImageNet
from keras.preprocessing import image as keras_prep  # used in vgg demo

from sklearn.utils import shuffle  # model training
from sklearn.preprocessing import LabelBinarizer

from keras.datasets import cifar10  # the dataset

"""
    Demo Keras networks
    VGG
"""
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input  # demo: use pretrained model as-is

# # ResNet
# model_resNet = ResNet50(weights='imagenet', include_top=False)  # 244x244 inputs
# VGG
model_vgg = VGG16(weights='imagenet')  # pretrained imageNet weights; To load without pretrained weights: "weights=None"

img_path = './small-traffic-set/img.png'

# The preprocessing technique must be the same with the one used to train the network for image net
img = keras_prep.load_img(img_path, target_size=(224, 224))  # vgg takes images 224x224
x = keras_prep.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = model_vgg.predict(x)  # Perform inference on our pre-processed image

print('Predicted:', decode_predictions(predictions, top=3)[0])  # Maps the prediction to the class name


"""
    Transfer learning lab:
    Demo Inception GoogleNet
"""
from keras.applications.inception_v3 import InceptionV3, preprocess_input  # demo: transfer learning

""" Model definition """
# Set a couple flags for training - you can ignore these for now
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet'  # 'imagenet' or None
preprocess_flag = True  # Should be true for ImageNet pre-trained typically

# Set a smaller input than the default "299x299x3" for InceptionV3 which will speed up training.
input_size = 139  # Keras v2.0.9 supports down to 139x139x3

# Keras requires us to set include_top to False in order to change the input_shape.
# By using "include_top=False", are dropped:
#   1.the final fully-connected layer with 1,000 nodes for each ImageNet class,
#   2.a Global Average Pooling layer.
# Also different ways instead of dropping are available:
#   (not drop the end and add new layers,
#   drop more layers than we will here, etc.).
#   Drop layers of a model with model.layers.pop() -> should check the actual layers
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(input_size, input_size, 3))

if freeze_flag:  # Freezing layers
    for layer in inception.layers:
        layer.trainable = False

inception.summary()  # review (print) all layers in the inception model

# We tell explicitly to the model which previous layer to attach to the current layer.
# e.g. x = Dropout(0.2)(inp)

""" Updating Model Input """
cifar_input = Input(shape=(32, 32, 3))  # Makes the input placeholder layer 32x32x3 for CIFAR-10

resized_input = Lambda(lambda image: tf.image.resize_images(  # Keras Lambda layer to re-size the input.
    image, (input_size, input_size)
))(cifar_input)  # Attach to cifar_input placeholder

# You will need to update the model name if you changed it earlier!
inp = inception(resized_input)  # Attach the input re-sizing layer to Inception model

""" Updating Model Output """
# Re-enter GlobalAveragePooling2D layer (removed with 'include_top=False'); connect it to the end of Inception.
out = GlobalAveragePooling2D()(inp)

# Create two new fully-connected layers using the Model API format.
out = Dense(512, activation='relu')(out)  # Attach to inception, along with ReLU; 512 or less nodes is a good idea.
predictions = Dense(10, activation='softmax')(out)  # Attach to last layer, with Softmax; 10 nodes CIFAR10 dataset classes

""" Create the full model, using the actual Model API """
model = Model(inputs=cifar_input, outputs=predictions)  # Create the model, assuming "predictions" as the final layer
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model

model.summary()  # Check the summary of this new model to confirm the architecture

# Use callbacks to avoid over-fitting; tooling callbacks bestModel, stop training early
save_path = './'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)  # if False every single epoch saves a model version
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)  # if accuracy is not decreased for delta for the last 'patience' epochs
# model.fit(callbacks=[checkpoint, stopper])  # train feeding-in the callbacks. Also other relevant data must be fed in

""" Train the model """
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)  # train
X_train, y_one_hot_train = shuffle(X_train, y_one_hot_train)
y_one_hot_val = label_binarizer.fit_transform(y_val)  # test
X_val, y_one_hot_val = shuffle(X_val, y_one_hot_val)

# Use only the first 10,000 images from the train set and the first 2,000 images from the test set for speed reasons
X_train = X_train[:10000]
y_one_hot_train = y_one_hot_train[:10000]
X_val = X_val[:2000]
y_one_hot_val = y_one_hot_val[:2000]

"""
Check: https://faroit.com/keras-docs/2.0.9/preprocessing/image/
you can also add additional image augmentation through this function, 
although we are skipping that step here so you can potentially explore it in the upcoming project.
Here also not using: image augmentation
"""
if preprocess_flag:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

# Train the model
batch_size = 32
epochs = 5
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1,
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)
