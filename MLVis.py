#!/usr/bin/env python
# coding: utf-8

# MLVis provides tools for interactively visualizing machine learning models. It is designed to support multiple levels of abstraction, but focuses on the TensorFlow/Keras layers API. The main functionality is a web-based graph visualization of an ML model's components, which integrates various plots and statistics about each layer with intuitive representations of the model's overall structure.

# In[158]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyvis
import json
import math
from functools import reduce
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
