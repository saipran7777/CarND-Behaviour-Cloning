**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/5epochs.png "Model Visualization"
[image2]: ./examples/15epochs.png "Model Visualization"
[image3]: ./examples/left.jpg "Recovery Image"
[image4]: ./examples/center.jpg "Recovery Image"
[image5]: ./examples/right.jpg "Recovery Image"
[image6]: ./examples/model.jpg "Recovery Image"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_car.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 containing a video of the car driving autonomously in a simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train_car.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths of 24 to 64 (train_car.py lines 66-70) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 64). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (train_car.py lines 71,77). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train_car.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In order to have a robust dataset, I drove the car for two laps on the simulator. I also drived the car in the reverse direction(after taking u turn) to collect more data which is unique.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was that these architectures are used to train real cars and have a good success rate in achieving it

My first step was to use a convolution neural network model similar to the one in Traffic Sign Classifier. I thought this model might be appropriate because when the model could recognize traffic signs from image data, I thought it could also detect road from the image and predict steering angle according to shape of the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it performs better on the validation set too. I modified the architecture similar to NVIDIA architecture later which improved model performance on the validation as well as in the simulator. I used Dropout layers in between to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like sharp turn near river side. To improve the driving behavior in these cases, I used more data near these spots

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train_car.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Grayscale image   							| 
| Lambda     	| Normalizes Input Data	|
| Cropping					|	Outputs 70x320x3									|
| Convolution 5x5     	| 2x2 stride, Valid padding	|
| RELU					|
| Convolution 5x5     	| 2x2 stride, Valid padding	|
| RELU					|	
| Convolution 5x5     	| 2x2 stride, Valid padding	|
| RELU					|	
| Convolution 5x5     	| Valid padding	|
| RELU					|	
| Convolution 5x5     	| Valid padding	|
| RELU					|	
| Dropout     	|	|
| Flatten  | |
| Dense		| Output = 100        									|
| Dense		| Output = 50       									|
| Dense		| Output = 10         									|
| Dropout     	|	|
| Dense			|  Output = 1       									|
|						|												|
 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get to center in case the car drift away from the track to either side. Then I repeated this process on track two in order to get more data points.

Instead of flipping images, I drove the car in the reverse direction, which solves the purpose.

After the collection process, I had 6954 number of data points. I then preprocessed this data by using lamda and croppping layer. The lambda layer was used to normalize the input data between -0.5 and +0.5. The Cropping layer was used then used to remove background , which was useless for training the model.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the following the image.

![alt text][image1]

I found the ideal number of epochs by first running it on large number of epochs and seeing the decrease of loss. The following graph shows the result.
![alt text][image2]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
