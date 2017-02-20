# **Behavioral Cloning**

---

## Project Description

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model.
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained convolution neural network. This file has a higher size than 100 MB and therefore can be downloaded from [here](https://drive.google.com/open?id=0B41pqIfTqFFobDNkTGJZS2pRcDA)
* writeup_report.md  for summarizing the results

####2. Submission includes functional code
Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip) and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py Model_Save/model.h5
```
After opening the simulator, the file `drive.py` connects to the simulator platform and provides the ability for the car to drive autonomously.
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. This file uses some predefined helper functions for image processing and defining the neural network architecture.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The neural network architecture implemented is the recently famous NVidia architecture consisting of convolutional network of 5x5 and 3x3 filter sizes and their respective depths of 24, 36 48 and 64 (keras_model.py lines 28-39). The model includes dense layers of depth 1164,100,50,10 and 1, with the last one being the normalized output steering angle.

The model includes ELU layers instead of the standard RELU, to introduce nonlinearity (code line 20). This was chosen because of the continuous nature of the derivative of ELU activation function near 0. This being a continuous regression problem, such type of activation function was better suited here. The input data as pixels is normalized in the model using a Keras lambda layer (code line 18) and also cropped using a Cropping2D function from keras.layers module.

####2. Attempts to reduce overfitting in the model

The initial attempts to reduce overfitting were done be using dropouts and L2 regularizers. However, balancing the training data by removing the ~0 degree turn images, flipping and augmenting the rest of the images led to a more robust training scheme and eventually, the neural network was able to capture the image details effectively without the use of dropouts and regularizers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
