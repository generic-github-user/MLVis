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


# In[7]:


tf.keras.backend.clear_session()


# In[8]:


layer_settings = dict(
    activation='relu'
)
model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape((28, 28, 1)),
    tf.keras.layers.Conv2D(4, 3, 1, **layer_settings),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 3, 1, **layer_settings),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dense(128, **layer_settings),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, **layer_settings)
])
model.summary()


# In[9]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[10]:


model.fit(train_images, train_labels, epochs=10)


# In[8]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# In[9]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


# In[11]:


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


# In[12]:


n = 23785
a = math.ceil(n**(1/2))
b = n // a
print(a*b)


# In[13]:


json.dumps(settings)


# In[23]:


import tensorflow.python.ops.numpy_ops.np_config as npcfg
npcfg.enable_numpy_behavior()


# In[ ]:


# for L in model.layers:
#     if hasattr(L, 'activation'):
#         L.activation = 


# In[55]:


# {'hierarchical': True}
vis = pyvis.network.Network(width=1000, height=1000, notebook=True, directed=True, layout=settings)
vis.set_options(json.dumps(settings, indent=4))
properties = [
#     ['activation', 'Activation'],
    ['units', 'Units'],
    ['shape', 'Shape'],
    ['input_shape', 'Input shape'],
    ['output_shape', 'Output shape'],
]
