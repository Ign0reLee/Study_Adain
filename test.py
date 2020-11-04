import os, sys, cv2, glob, argparse

import numpy as np
import random as rd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as k

from net import *
from tensorflow.keras.layers import  *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

# Set GPU Operation Growth
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(usage="train.py --ct ContentDirectory --st StyleDirectory [Option]")
parser.add_argument("--ct", help="Content File You Wants")
parser.add_argument("--st", help="Style File You Wants")
parser.add_argument("--load_dir", help="Save Check points, You Wants", default=os.path.join(".","CheckPoints"))
parser.add_argument("--load_index", help="Save Check points, You Wants", default=-1)
parser.add_argument("--out_dir", help="Save Output Files Path You Wants", default=os.path.join(".","Outputs"))

args = parser.parse_args()

Content_Path   = args.ct
Style_Path     = args.st
Load_Path      = args.load_dir
Output_Path    = args.out_dir
Load_index     = args.load_index

if not os.path.exists(Output_Path):  os.mkdir(Output_Path)  
if not os.path.exists(Load_Path):    raise TypeError("Please Check the Load Weight Path")
if not os.path.exists(Content_Path): raise TypeError("Please Check the Content Images Path")
if not os.path.exists(Style_Path):   raise TypeError("Please Check the Style Images Path")


# Weight File Load

Weight_Files = glob.glob(os.path.join(Load_Path, "*"))
Weight_File  = Weight_Files[Load_index]

# Define Layer
Loss_Layer = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1'] 

# Define Output File Name

file_name = "AdaIN Output"


# Make Model and Load
encoder = pre_vgg(Loss_Layer)
model = AdaIn_Transfer()
model.load_weights(Weight_File)


#Load Image End Make New File

c = load_image(Content_Path)
s = load_image(Style_Path)

v_c = encoder(np.expand_dims(c, axis=0))
v_s = encoder(np.expand_dims(s, axis=0))

output = model(v_c[-1], v_s[-1], training=False)[0].numpy().astype(np.uint8)
plt.imsave(os.path.join(Output_Path,file_name+".jpg"),output) 
