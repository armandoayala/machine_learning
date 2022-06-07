#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import dependencies
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds


# In[3]:


#Check version tensorflow
print(tf.__version__)


# In[4]:


# Load images dataset
# https://www.tensorflow.org/tutorials/load_data/images
# To read train_images we must download the directory train_images from
# https://www.kaggle.com/competitions/hotel-id-to-combat-human-trafficking-2022-fgvc9/data
# After downloaded the images, we must unzip the file
batch_size = 32
img_height = 180
img_width = 180
data_dir="D:\\Armando\\MLMentoria\\train_images"


# In[5]:


# Using keras utility to build dataset and we will inferred labels. Also we resize the images to standard size 180 x 180
# Note each sub-dir in train_images will be a label, each label is an id of a hotel 


# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
# Build datset to train model. We will use 80% of images to train the model
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[8]:


class_names = train_ds.class_names
print(class_names)


# In[6]:


# Build datset to validate model. We will use 30% of images to validate the model
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[9]:


# Visualize some images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[ ]:




