from datetime import time

import numpy as np
import tensorflow as tf
import PIL
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dataset
import sys

def make_dblock(model, layers):
    k = (3, 3) 
    g = layers    
    model.add(tf.keras.layers.Conv2D(g, k, strides=(1, 1), use_bias=False, kernel_initializer='he_uniform',  padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.1)) 
    model.add(tf.keras.layers.MaxPool2D())
    return model 



def make_model(image_size):
    w = image_size
    g = 32
   

    model = tf.keras.Sequential()
    k = (3, 3)
    model.add(tf.keras.layers.Reshape([w, w, 3], input_shape=(  w, w, 3))) 
    print("D ",model.output_shape)
    model.add(tf.keras.layers.Conv2D(g, k, strides=(1, 1), padding='same', use_bias=False,    kernel_initializer='he_uniform',    input_shape=[  32, 32, 3]))
 
    make_dblock(model, g) 
    print("D ",model.output_shape)
    assert model.output_shape == ( None, w//2, w//2,  g )  

    make_dblock(model, g*2)
    print("D ",model.output_shape)
    assert model.output_shape == ( None, w//4, w//4, g*2)        

    make_dblock(model, g*4)
    print("D ",model.output_shape)
    assert model.output_shape == ( None,  w//8, w//8, g*4)  

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(g*2, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax ))
    return model

def getdata(filename,dsize):     
    im = PIL.Image.open(filename).convert("RGB")
    imm = im.resize([dsize,dsize])
    data =  np.asarray(imm)*(1.0/255.0)    
    return  2.0*np.array(data) - 1.0


IMG_SIZE= 64 

model = make_model(IMG_SIZE)
model.load_weights("classificador_10/classificador")


 
#image = getdata(".\\AID\\Forest\\forest_33.jpg", IMG_SIZE )
image = getdata(sys.argv[1], IMG_SIZE )  


p = np.expand_dims(image, axis=0)
y=  model.predict( p  )
print(y)
 

 