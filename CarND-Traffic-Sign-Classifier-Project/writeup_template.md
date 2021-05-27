# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic-signs-data/data_distr_initial.png "Data Visualization"
[image2]: ./traffic-signs-data/preproc_orig.png "Sample Initial Image"
[image3]: ./traffic-signs-data/preproc_gray.png "Gray converted initial Image"
[image4]: ./traffic-signs-data/preproc_gray_norm.png "Normalized gray image"
[image5]: ./demo/demo_signs.png "Traffic signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

As the number of the data in some categories is bigger than in some others, I tried to generate additional validation and training data,
by rotating the initial images by +-90 and 180 degrees and applying random Gaussian blur with random window sizes 1 3 or 5. 
It turned out that probably this additional data led to strong over-fitting, because the network, with several layers (from 5 to 9) and 
filter depths (32-256), couldn't achieve validation accuracy more than 40%. A question is, 
what kind of changes would be adequate for extending the dataset. According to the paper of Sermanet referenced in the course
blurring should be adequate for differentiating the images.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as the advertised accuracy of the network of 89% can be achieved
with grayscaled images. A positive effect of gray-scaling is that the dimensions of the network can be kept low, as only two dimensions per image need to be processed.
Another positive aspect, is that the size of the preprocessed data is much lower in for grayscale images.
I guess that a quantity of almost 2GB is produced from an initial size of size 400MB of loaded images, after normalization, because of the float datatype of the normalized images. 

As a second preprocessing step, I normalized the image data because a valid dataset with zero mean and same distribution is necessary for training a neural network.
For the normalization, I calculated applied the operations "(x -128.)/128" to every pixel x. 

Another method I dried using mean and standard deviation using opencv as "new_pixel_val = (old_pixel_val - mean)/std", did
work for the gray-scaled images as it led in some cases to division by zero. 

Below a sample image is shown, before, after grayscaling, and after the normalization...

![alt text][image2]
![alt text][image3]
![alt text][image4]

The data numbers in the processed set are the following ... 
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 24x24x64 	|
| RELU					|												|
| Dropout				| Retain probability 0.9    					|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 22x22x64 	|
| RELU					|												|
| Dropout				| Retain probability 0.7    					|
| Max pooling	      	| 2x2 stride,  outputs 11x11x64 				|
| Flattening            |                                               |
| Fully connected		| Input 7744. Output 1290               		|
| RELU					|												|
| Dropout				| Retain probability 0.6    					|
| Fully connected		| Input 1290. Output 258                   		|
| RELU					|												|
| Dropout				| Retain probability 0.6    					|
| Fully connected		| Input 258. Output 43                     		|
| Softmax				|           									|
| Loss 					| Softmax + Regularization term					|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tried AdamOptimizer and RMSPropOptimizer optimizers, but I left the first one as suggested in the course. 
Although the both deliver equivalent good results, the first one seems to be robuster. The learning rate must be adjusted a bit, depending on which optimization method is used. 

The weights are initialized with the He-normal distribution from the keras package, as suggested in the Stanford course, 
because they seem to give a better starting point for the training process.

Additionally, a dropout layer was inserted after several layers to increase the stability of the network.

At first, I decided to leave out max-pooling because in both the course and stanford lecture, is described 
that it is a very hard method to reduce the parameters and deteriorates the image quality. However, I saw that max-pooling
applied every some convolutions gives better results, than say a convolution with stride 2 and 'SAME' padding. 
I also experimented with batch normalization, but this gave worse results, and I left it out.

Other features of the network, are the l2 regularization of the weights, and a decaying learning rate (about 50% in each epoch).

I searched with bigger and smaller steps around the given batch sizes and epochs.
A question is if it should really matter which batch size is used, or if we take a value (e.g. 128) and experiment with the rest parameters.
If it does matter, which should be a good method to find the appropriate batch size?  

Once I concluded to the batch size, I tried several values for the learning rate between 0.01 and 0.0005 and I found that values around 0.001 give the best results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.948 
* test set accuracy of 0.937

If an iterative approach was chosen:
* I started with the LeNet model and the available data, which was already distributed in train-evaluation-test groups opposite to what is said in the video.
  * The model gave less than 80% success in the validation data, for color images.
* I tried to extend the dataset, but what I reached was to break the data, end experinet in the false direction.
* It seems that with only the two used layers and by extending the filter depth the network is able to reach 95% of accuracy.
* I chose to add one more layer with filter size 3x3.
* During the exploration of the broken dataset, I implemented some patterns described in the Stanford lecture 
  (INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC), but with this dataset nothing worked. 
* Another option whould be to use the ResNet, but I avoided it because of its size, fearing that it lead to overfitting.
* In the final implementation I used max-pooling only once, because it is described as having a negative effect in the learning process 
  (however they are used in almost all examples!?). A 2x2 stride instead did not work that good.
* Finally, I fine-tuned the step size, and the keep rate for drop-out, and integrated the regularization, which during the course of adaptations I had left out.
* For the drop-out, when used in the convolution layers, it leaks weights because of the convolution, but I kept it as it improved the results.
* I found that decreasing the size of the 2 first layer filters to 3, does not necessarily improves the performance, so I left it to 5.  
* An interesting observation is that the network's performance is a bit unstable. As the model is retrained more times, 
  although the performance on validation data is in general over 93%, the randomization of batches led to performances up to 95%.
  Also the convergence rate in along the epochs varies significantly. This can be due to the randomization of the images, or a marginal learning rate.
* Every training session, was started fresh, and no data was reloaded, when started a training session. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![alt text][image5]

I picked out some random images, which the network had no problem to classify correctly.
The first and third images might be difficult to classify because they are dark. The rest images should be easy to identify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 50        | Speed Limit 50         						| 
| No Vehicles  			| No Vehicles	  							    |
| Bicycles crossing		| Bicycles crossing								|
| End of no passing		| End of no passing    							|
| Roundabout mandatory  | Roundabout mandatory 							|
| Priority road			| Priority road							        |
| Yield					| Yield											|

The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. 
This lies above the accuracy on the test set of 93,7%. I would expect that some rare signs like "Pedestrians" would be more difficult for the network to identify.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top-5 probabilities are written, in the image_stats.txt in the demo folder.

category: 31 prediction: [[31 21 25 29  2]]
stats: [[55.967  4.522  3.740  2.093  2.046]]%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .56         			| Wild animals crossing   									| 
| .04     				| Double curve 										|
| .04					| Road work											|
| .02	      			| Bicycles crossing					 				|
| .02				    | Speed limit (50km/h)     							|


category: 12 prediction: [[12 40 11  7 30]]
stats: [[90.068  2.480  0.796  0.738  0.730]]%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .90         			| Priority road   									| 
| .02     				| Roundabout mandatory										|
| .08					| Right-of-way at the next intersection											|
| .07	      			| Speed limit (100km/h)					 				|
| .07				    | Beware of ice/snow     							|

category: 2 prediction: [[2 1 3 4 5]]
stats: [[8.794 8.162 4.403 3.885 3.872]]%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .087         			| Speed limit (50km/h)   									| 
| .081     				| Speed limit (30km/h) 										|
| .044					| Speed limit (60km/h)											|
| .039	      			| Speed limit (70km/h)					 				|
| .039				    | Speed limit (80km/h)     							|

category: 41 prediction: [[41 32 42 20 12]]
stats: [[64.173  5.564  4.075  3.215  2.476]]%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| End of no passing   									| 
| .05     				| End of all speed and passing limits 										|
| .04					| End of no passing by vehicles over 3.5 metric tons											|
| .03	      			| Dangerous curve to the right					 				|
| .02				    | Priority road      							|

category: 15 prediction: [[15 12  3  5  9]]
stats: [[69.233  3.224  2.541  2.374  1.843]]%

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| No vehicles  									| 
| .03     				| Priority road										|
| .03					| Speed limit (60km/h)										|
| .02	      			| Speed limit (80km/h)					 				|
| .02				    | No passing      							|


category: 13 prediction: [[13 35  9 34 15]]
stats: [[9.835e+01 5.544e-01 2.814e-01 1.266e-01 6.941e-02]]%

category: 40 prediction: [[40  7 30 42 11]]
stats: [[33.902  5.832  4.714  4.438  3.181]]%


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


