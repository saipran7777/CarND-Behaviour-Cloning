
# coding: utf-8

# In[1]:


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


train = pd.read_csv('train/driving_log.csv',header=None)
images = []
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
    images.append(img3)
for i in train[3]:
    correction = 0.2
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


model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))

model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(84,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))


# In[ ]:


model.compile(loss='mse',optimizer='Adam')
model.fit(X_data,y_data,validation_split=0.2,shuffle=True,epochs=2)
model.save('model.h5')

