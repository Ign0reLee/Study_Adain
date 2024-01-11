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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
# Set GPU Mirror Strategy
strategy = tf.distribute.MirroredStrategy()

# Parsing Part

parser = argparse.ArgumentParser(usage="train.py --ct ContentDirectory --st StyleDirectory [Option]")
parser.add_argument("--ct", help="Content Directory You Wants")
parser.add_argument("--st", help="Style Directory You Wants")
parser.add_argument("--lr", help="Learning Rate You Wansts", default=1e-4)
parser.add_argument("--lr_decay", help="Learning Rate Decay", default=5e-5)
parser.add_argument("--cw", help="Content Weight You Wants", default=1.0)
parser.add_argument("--sw", help="Style Weight You Wants", default=1e-2)
parser.add_argument("--epochs", help="How Many Epochs You Wants", default=2)
parser.add_argument("--batch_size", help="Batch Size You Wants", default=8)
parser.add_argument("--visual_dir", help="Step Visualization Path", default=os.path.join(".", "Visualization_Training"))
parser.add_argument("--save_dir", help="Save Check points, You Wants", default=os.path.join(".","CheckPoints"))


args = parser.parse_args()


Content_Path   = args.ct
Style_Path     = args.st
Save_Path      = args.save_dir
Visual_Path    = args.visual_dir
lr             = float(args.lr)
lr_decay       = float(args.lr_decay)
batch_size     = int(args.batch_size)
Content_Weight = float(args.cw)
Style_Weight   = float(args.sw)
epochs         = int(args.epochs)

if not os.path.exists(Save_Path):    os.mkdir(Save_Path)
if not os.path.exists(Visual_Path):  os.mkdir(Visual_Path)
if not os.path.exists(Content_Path): raise TypeError("Please Check the Content Images Path")
if not os.path.exists(Style_Path):   raise TypeError("Please Check the Style Images Path")


# Define Train Step
with strategy.scope():
    @tf.function
    def train_step(ct_features, st_features):
        
        with tf.GradientTape() as tape: 
            output = model(ct_features[-1], st_features[-1], training=True)

            output_features = encoder(output)
            loss = model.AdaIN_Loss(output_features, ct_features, st_features, Style_Weight)

        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        tf.print("에포크 : ",epoch,  " 스텝 : ", step ,", Loss : " , loss, output_stream=sys.stderr)

    @tf.function
    def distributed_train_step(ct_inputs, st_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,args=(ct_inputs, st_inputs))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


# Define Layer
Loss_Layer = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1'] 


# Find all Images in the Image Path

Content_Path = glob.glob(os.path.join(Content_Path, "*"))
Style_Path   = glob.glob(os.path.join(Style_Path, "*"))

#Slice Same Size Between Style Images and Content Images



Train_Content_Path, Validation_Content_Path = Content_Path[:-1], Content_Path[-1]
Train_Style_Path,   Validation_Style_Path   = Style_Path [:-1],  Style_Path [-1]


# Define Train Model

with strategy.scope():
    
    encoder = pre_vgg(Loss_Layer)
    model = AdaIn_Transfer()
    lr_schedule = k.optimizers.schedules.InverseTimeDecay(lr, decay_steps=1, decay_rate=lr_decay, staircase=False)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

# Define Validation Data

val_c = load_image(Validation_Content_Path)
val_s = load_image(Validation_Style_Path)
plt.imsave(os.path.join(Visual_Path,"Original_Content.jpg"), val_c.astype(np.uint8))
plt.imsave(os.path.join(Visual_Path,"Original_Style.jpg"), val_s.astype(np.uint8))

v_c = encoder(preprocess_input(np.expand_dims(val_c, axis=0)))
v_s = encoder(preprocess_input(np.expand_dims(val_s, axis=0)))



# Start Train

step = 0

for epoch in range(epochs):
    
    #Make New Traing Batches
    rd.shuffle(Train_Content_Path)
    rd.shuffle(Train_Style_Path)

    Train_Content_Path = Train_Content_Path[:len(Train_Style_Path)]
    max_value = len(Content_Path) // batch_size * batch_size

    Train_Content_Path = Train_Content_Path[:max_value]
    Train_Style_Path   = Train_Style_Path[:max_value]

    Content_batch = data_batch(Train_Content_Path)
    Style_batch  = data_batch(Train_Style_Path)

    with strategy.scope():
    
        for ct_img, st_img in zip(Content_batch, Style_batch):
            
            #Make Distributed data
            ct_data = tf.data.Dataset.from_tensors(ct_img).batch(batch_size)
            ct_data = strategy.experimental_distribute_dataset(ct_data)

            st_data = tf.data.Dataset.from_tensors(st_img).batch(batch_size)
            st_data = strategy.experimental_distribute_dataset(st_data)

            for ct, st in zip(ct_data, st_data):
        
                ct_features = encoder(ct)
                st_features = encoder(st)
                distributed_train_step(ct_features,st_features)
                
            #Validate Every 2000 Step
            if( step % 2000 ==0): 

                val_image = model(v_c[-1], v_s[-1], training=False)[0].numpy().astype(np.uint8)
                file_name = str(epoch)+"_"+str(step)
                cv2.imwrite(os.path.join(Visual_Path,file_name+".jpg"),val_image) 
                model.save_weights(os.path.join(Save_Path, file_name+"_Adain.ckpt")) 

            step +=1

    #Validate Evry Epoch
    val_image = model(v_c[-1], v_s[-1], training=False)[0].numpy().astype(np.uint8)
    file_name = str(epoch)+"_"+str(step)
    cv2.imwrite(os.path.join(Visual_Path,file_name+".jpg"),val_image) 
    model.save_weights(os.path.join(Save_Path, file_name+"_Adain.ckpt")) 
