
# coding: utf-8

# In[1]:

# Importing Dependencies
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D,Dropout


# In[2]:

# Input Data
train = pd.read_csv('train/driving_log.csv',header=None)
images = [] # Initializing empty variables
measurements = []


# In[3]:


for i,j in train.iterrows():
    x1 = j[0].split('/')[-1]
    x2 = j[1].split('/')[-1]
    x3 = j[2].split('/')[-1]
    img1 = mpimg.imread('../data/IMG'+x1)
    img2 = mpimg.imread('../data/IMG'+x2)
    img3 = mpimg.imread('../data/IMG'+x3)
    images.append(img1)
    images.append(img2)
    images.append(img3) # adding to list
for i in train[3]:
    correction = 0.2 # assumed correction factor to adjust the bias in steering prediction
    measurements.append(float(i))
    measurements.append(float(i)+correction) # left 
    measurements.append(float(i)-correction) # right


# In[4]:


X_data = np.array(images)
y_data = np.array(measurements)


# In[5]:


shape = X_data[1].shape
print(shape)


# In[ ]:

# Convolution Neural Network - Nvidia Architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))


# In[ ]:

# Training the model
model.compile(loss='mse',optimizer='Adam')
model.fit(X_data,y_data,validation_split=0.2,shuffle=True,epochs=2) # split train data 
model.save('model.h5')

