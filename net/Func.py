import os, cv2

import numpy as np

import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

def pre_vgg(layers):
        
    vgg = vgg19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers]
    model = k.Model([vgg.input], outputs)

    return model

def load_image(path_to_img):
    
    
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256)).astype(np.float32)
    

    if not (len(img.shape) == 4):
        img = np.expand_dims(img, 0)


    return img

def make_image_minbatch(path_to_img):

    batches = load_image(path_to_img[0])

    for path  in path_to_img[1:]:
        batches = tf.concat([batches, load_image(path)], axis=0)
        
    return preprocess_input(batches)

def data_batch(path_to_img, batch_size=8):
    
    for index in range(0, len(path_to_img), batch_size):
        yield make_image_minbatch(path_to_img[index:index + batch_size])

