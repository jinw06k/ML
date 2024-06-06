#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist


# In[2]:


(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

img = plt.imshow(train_X[10])
plt.colorbar(img) 
plt.show()


# In[3]:


train_X = train_X/256
test_X = test_X/256

img = plt.imshow(train_X[10])
plt.colorbar(img) 
plt.show()


# In[4]:


import keras
from keras import layers


# In[6]:


model = keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ]
)


# In[7]:


learning_rate_e = 0.001


# In[8]:


model.compile(loss='SparseCategoricalCrossentropy', 
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate_e),
              metrics=['accuracy'])


# In[9]:


model.fit(train_X, train_y, epochs=3)


# In[10]:


loss, accuracy =  model.evaluate(test_X, test_y)


# In[12]:


predictions = model.predict([test_X])


# In[13]:


plt.imshow(test_X[0])
plt.show()
print(np.argmax(predictions[0]))


# In[14]:


plt.imshow(test_X[94])
plt.show()
print(np.argmax(predictions[94]))


# In[15]:


plt.imshow(test_X[123])
plt.show()
print(np.argmax(predictions[123]))


# In[ ]:




