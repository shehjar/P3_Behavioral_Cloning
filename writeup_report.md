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
* model.h5 containing a trained convolution neural network. This file has a higher size than 100 MB and therefore can be downloaded [here](https://drive.google.com/open?id=0B41pqIfTqFFobDNkTGJZS2pRcDA)
* writeup_report.md  for summarizing the results

####2. Submission includes functional code
Using the Udacity provided [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip) and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.json
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

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). The learning rate was however initialized as 0.001

####4. Appropriate training data

The training data included a skewed ratio of more than half of the training samples as close to 0 angle steering. The data was balanced by removing the images and their corresponding steering angles which fell between -0.01 and 0.01. Later during fit generator, the left, right and center images and their flipped images were given as an input to the neural network model.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was for the neural network to grasp the minute details of the road that represented road curvature, so that it could be applied to any other terrain.

My first step was to use a convolution neural network model similar to the NVidia architecture, since they used the left and the right camera images for training their model and I thought this model might be appropriate because it was deep and with two different sets of filters, it was faster at evaluating high dimensional pixel data without using maxpooling techniques.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. On most occasions, the validation loss was less than training loss. The decrease in value of the loss was proportional on both the datasets at each epoch. On the best working model configuration both the losses stagnated into a semi-constant value.

However to observe overfitting, one really had to run the simulator and see the result. Initially it was difficult to ascertain which condition implied what result. An extremely hard turn on a straight road might be a result of overfitting and missing a valid turn during the course of simulation might be a result of underfitting. In the initial stages of the development of the model, I had found myself in both the categories and I tried using dropouts and L2 regularizers, changed its parameters to stay between the case of over and under fitting.

Later, I reformed my training data to include only tangible turns such that the steering angle for an example training data should be greater than 0.01 or less than -0.01. The respective images were augmented in terms of translation, brightness and shadows and the classifier was fed with a more balanced dataset. Through this, I was able to avoid dropouts and regularizers as the model worked with a more varied dataset.

The images color space was later changed to HLS color space and that greatly increased the model efficiency.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, although with some jittery motion on certain occasions.

####2. Final Model Architecture

The final model architecture (keras_model.py) consisted of a convolution neural network with two 5x5 and 3x3 filter sizes

Here is a snippet of the layers of the NVidia neural network architecture
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 103, 320, 3)   0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 103, 320, 3)   0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 50, 158, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 50, 158, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 23, 77, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 23, 77, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 37, 48)    43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 10, 37, 48)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 8, 35, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 8, 35, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 6, 33, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 6, 33, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 12672)         0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          14751372    flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        elu_7[0][0]                      
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         elu_8[0][0]                      
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          elu_9[0][0]                      
====================================================================================================
Total params: 15,004,791
Trainable params: 15,004,791
Non-trainable params: 0
____________________________________________________________________________________________________
```

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
