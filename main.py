from datetime import time

import numpy as np
import tensorflow as tf

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dataset


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


IMG_SIZE= 64 

model = make_model(IMG_SIZE)
model.compile(optimizer='adam',   loss='categorical_crossentropy',   metrics=['accuracy'])




training_dataset = dataset.getImageDataSetForest(IMG_SIZE) 
y_label = [] 
training_dataset_full =[]

print("Loading DataSet 1/3")
for q in dataset.getImageDataSetForest(IMG_SIZE):
    training_dataset_full.append(q)
    y_label.append([1.0, 0, 0])  

print("Loading DataSet 2/3")    
for q in dataset.getImageDataSetSparseResidential (IMG_SIZE):
     training_dataset_full.append(q)
     y_label.append([0.0, 1.0, 0])  

print("Loading DataSet 3/3")
for q in dataset.getImageDataSetBareLand (IMG_SIZE):
    training_dataset_full.append(q)
    y_label.append([0.0, 0, 1.0])  

train_images = tf.convert_to_tensor(training_dataset_full,dtype=tf.float32)
train_labels = tf.convert_to_tensor(y_label,dtype=tf.float32)

print(train_labels.shape)
print(train_images.shape)
model.fit(train_images, train_labels, epochs=100)



model.save_weights("classificador_{}/classificador".format(10))

 