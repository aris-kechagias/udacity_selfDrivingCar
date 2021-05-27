#**Behavioral Cloning**

---

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


[//]: # (Image References)

[image1]: ./center_2021_02_02_10_42_06_349.jpg
[image2]: ./center_2021_02_02_10_42_28_925.jpg
[image3]: ./center_2021_02_02_10_42_41_434.jpg
[image4]: ./center_2021_02_02_10_42_53_938.jpg
[image5]: ./center_2021_02_02_10_48_18_761.jpg
[image6]: ./center_2021_02_02_10_48_40_137.jpg
[image7]: ./center_2021_02_02_10_48_48_201.jpg
[image8]: ./center_2021_02_02_10_49_12_486.jpg
[image9]: ./center_2021_02_02_10_49_32_238.jpg
[image10]: ./modified_nvidia_pipeline.png

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* weights.05-0.03.hdf5
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is the implementation of the nvidia network for self-driving car [network](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/),
although it may differentiate from the original implementation as it will be described below.
It consists of a first cropping layer, which cuts out the top (63 pixels) and bottom (27 pixels) part of the image, 
which could corrupt the learning process. This is followed by the second normalization layer, where the input images
are modified to have zero mean and small variance. The third layer resizes the images to conform with the nvidia network.

After resizing 3 convolutional layers with filter kernel and stride 2 follow. The number of filters in each layer is
24, 36 and 48 respectively. They are followed from two more convolutional layers with kernels 3x3 and 64 filters. Those
two layers use no stride.

Although in the paper no activation function for the convolutional layers is mentioned, I included relu activation.
I experimented with including l2 regularization in the convolution layers as well, but the loss reaced only 0.5 in 
contrast to 0.3 without so, I did not use regularization.

After the convolutional layers, a flattening layer follows. The resulting neurons are 1152, however in the paper the number 1164
for the first dense layer is referred. Thus, I upscaled the neurons of the flattening layer to 1164. After the first dense
layer four more follow with outputs 100, 50, 10 and 1 as described in the paper. For the first two dense layers, I included
dropout layers, although nothing similar is mentioned by nvidia.

Another difference to the nvidia paper, is that they convert the input image to YUV. I avoided this as the drive.py script
uses PIL.Image to load the images in RGB, and I did not know if this choice was imposed by a potential processing speed
limitation when using opencv.

A further potential difference is, that nvidia mentions that "the YUV planes were fed to the network" which could mean
that the network in the original form could receive 2D images instead of 3D and somehow average or weight the predictions
for the 3 planes. This option was not implemented, as the network works well with RGB images already. 

In the generator function the images are opened with Image.open() to align with the the drive.py. Furthermore, the side
images were corrected by the constant of 0.17 compared to the center image.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after the two dense layers, where the number of parameters
is still big. The drop ratio is chosen rather modestly as 0.2.

The model was trained and validated on different data sets to ensure that the model was not overfitting. These include
the provided dataset from udacity, two forward and two backward records from the first track, as well as alignment records
towards the center of the same track. Additionally, two froward and one backward records were taken from the second track.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, 
recovering from the left and right sides of the road, and records from both tracks.

For details about how I created the training data, see the next section.

### Architecture and Training Documentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning, as I noticed from the previous project
that it can be daunting to conclude in a working structure. Specifically, it can be absolutely disorientating whether
the network does not serve its purpose, or if the data are corrupted. Thus I used the nvidia network, which is designed
specifically for driving.

The final step was to run the simulator to see how well the car was driving around the track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

As the used network is a modified nvidia cnn, a modified visualization from the corresponding nvidia
web page is presented below.

![alt text][image10]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded multiple forward and backward drives from both tracks, together with
short correction drives for the first track. The recorded data can be downloaded from [dropbox](https://www.dropbox.com/sh/ffqzzi18b0n4r3b/AAB-f-MbQLPcdZTrcpy9X5T8a?dl=0)
The recordings are described above under [Attempts to reduce overfitting in the model](). As I recorded enough data, 
I decided to avoid augmenting the data set.

Here are some samples of the collected images for both tracks:\
![alt text][image1]![alt text][image2]
![alt text][image3]![alt text][image4]
![alt text][image6]![alt text][image7]
![alt text][image8]![alt text][image9]

Correction maneuver sample:\
![Correction maneuver][image5]



After the collection process, I had 60306 number of data points. I also randomly shuffled the data set and put 30% of 
the data into a validation set. The validation set helped determine if the model was over or under fitting. 
I stopped the training process after 5 epochs, as the loss was almost stabilized around 0.3. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Although, this project has very good descriptions and resources, I found that the training in the workspace had the 
problem of creating a model.h5 file which is not compatible with my distribution. To solve this, I loaded locally the 
model parameters which were saved in the workspace after the training. Finally, I minimally modified the drive.py to consider 
tensorflow as custom object in line 127 to be able to include the resizing layer.


