# **Behavioral Cloning** 

## Project Report
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./before_flipped.png "Before flipped"
[image2]: ./flipped.png "Flipped"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes (with strides size = 2x2) and 3x3 filter sizes (with strides size = 1x1). The depths are between 24 and 64 (model.py lines 143-147) 

The model includes RELU layers to introduce nonlinearity (code line 143-147), and the data is normalized in the model using a Keras lambda layer (code line 142). 

#### 2. Attempts to reduce overfitting in the model

The model didn't use any drop out or regularization method to avoid overfitting and it has a robust performance.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 158-166). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 158).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, where images from left camera could be regarded as the car drifting towards left of the road and images from right camera could be regraded as the car drifting towards right of the road. We need to set the offset parameter to compensate for the corresponding steering angles (line 26)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was combining a bunch of different linear and nonlinear functions to build up a function approximator (use transfer learning method) which could extract the key info in the image and output the steering angle.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because LeNet is proved to have the capacity of extracting key features from the image during its classification task. Hence I only need to adjust its end layers to match the size of my output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on both the training set as well as the validation set (around 0.25). This implied that the model was underfitting. 

To combat the underfitting, I modified the model so that more training epochs are added.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially the one when the car exit the bridge and encounter a big turn. To improve the driving behavior in these cases, I uses the NVIDIA end-to-end CNN.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 140-152) consisted of a convolution neural network with the following layers and layer sizes:

Input: 160x320x3

Cropping layer: 90x320x3

Normalization layer: 160x320x3

Convolutional layer: 78x163x24

RELU layer: 78x163x24

Convolutional layer: 37x80x36

RELU layer: 37x80x36

Convolutional layer: 17x38x48

RELU layer: 17x38x48

Convolutional layer: 15x36x64

RELU layer: 15x36x64

Convolutional layer: 13x34x64

RELU layer: 13x34x64

Flatten: 28288

Fully Connected: 100

Fully Connected: 50

Fully Connected: 10

Fully Connected: 1

#### 3. Creation of the Training Set & Training Process

I only use the given data set.

To augment the data set, I flipped images and angles thinking that this would  For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

I also use left and right camera image to augment the data set. But it is kinda tricky when adjusting the compensating sterring angle. Initially I set `correction = 1.25` and the car's behavior is too drastic. After rounds of tuning, I found the optimal value is `correction = 0.5`.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the low and converged loss error in both training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
