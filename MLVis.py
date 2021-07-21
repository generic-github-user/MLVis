#!/usr/bin/env python
# coding: utf-8

# MLVis provides tools for interactively visualizing machine learning models. It is designed to support multiple levels of abstraction, but focuses on the TensorFlow/Keras layers API. The main functionality is a web-based graph visualization of an ML model's components, which integrates various plots and statistics about each layer with intuitive representations of the model's overall structure.

# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyvis
import json
import math
from functools import reduce
import webbrowser


# In[2]:


# From https://stackoverflow.com/a/6800214/10940584
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


# In[3]:


factors(327)


# In[4]:


# todo: allow model editing (?)


# In[ ]:


# Sample model is based on code from TensorFlow/Keras image classification tutorial at https://www.tensorflow.org/tutorials/keras/classification


# In[5]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


train_images = train_images / 255.0

test_images = test_images / 255.0
settings = {
  "edges": {
    "color": {
      "inherit": True
    },
    "smooth": False
  },
  "layout": {
    "hierarchical": {
      "enabled": True,
      "levelSeparation": 260,
      "nodeSpacing": 255,
      "treeSpacing": 195,
      "sortMethod": "directed"
    }
  },
  "physics": {
    "enabled": False,
    "hierarchicalRepulsion": {
      "centralGravity": 0
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  }
}


# In[192]:


n = 23785
a = math.ceil(n**(1/2))
b = n // a
print(a*b)


# In[193]:


json.dumps(settings)
