# Author Luís Cunha 1200650@isep.ipp.pt
# 
# Model: "Mobilenetv2"
# Found 1056 files belonging to 2 classes.
# Using 212 files for training.
# Using 52 files for validation.
# Total params: 2,265,666
# Trainable params: 2,228,994
# Non-trainable params: 36,672

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers, applications
from keras.models import Sequential
from keras.models import load_model

tf.autograph.set_verbosity(0)

import pathlib
data_dir = pathlib.Path('./img/')
image_count = len(list(data_dir.glob('./img/*/*.jpg')))
print(image_count)

batch_size = 8
img_height =224
img_width = 224
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.9,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.05,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)])

num_classes = len(class_names)

from keras.applications import MobileNetV2
#EfficientNetB0, EfficientNetV2B0, ResNet50, Xception, MobileNetV2

img_augmentation = Sequential([
        layers.RandomRotation(factor = 0.15),
        layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1),
        layers.RandomFlip()],
    name = "img_augmentation")

def build_model():
    inputs = layers.Input(shape = (img_height, img_width, 3))
    x = img_augmentation(inputs)
    x= layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))(x)
    model = keras.applications.MobileNetV2(classes = 2, include_top = False, weights = None, input_tensor = x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.1
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(2, activation = "softmax", name = "pred")(x)
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    #loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    model.compile(
        optimizer=optimizer, 
        loss=loss, 
        metrics=["accuracy"])
    model.summary()
    return model

model = build_model()
    
epochs=5
# import from saved model?
# model= load_model('dfu_MobileNetV2.h5')

history = model.fit(  train_ds,  validation_data=val_ds,  epochs=epochs )
model.save('dfu_MobileNetV2.h5')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
file1 = open("mobilenet2.txt", "a") # write loss+acc to cvs file
for i in  range(epochs):
    file1.write(str(loss[i]) + ","+str(acc[i])+"\n")
file1.close()

def x_predict (test_img_path):
    img = tf.keras.utils.load_img(
        test_img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("{} predicted to be: {} at {:.2f} percent confidence."
        .format(test_img_path, class_names[np.argmax(score)], 100 * np.max(score)))
    return np.argmax(score)

def pred():
    numfails=0
    for i in range (1,440):
        mstr= './img/Abnormal/' + str(i) + '.jpg'
        # print (mstr)
        if x_predict(mstr) == 1:
            numfails += 1
    print        
    print ("Number of failed predictions: " +str(numfails))
    print(numfails*100/440)
    falsepos = numfails

    numfails=0
    for i in range (1,533):
        mstr= './img/Normal/' + str(i) + '.jpg'
        # print (mstr)
        if x_predict(mstr) == 0:
            numfails += 1
    print        
    print ("Number of failed predictions: " +str(numfails))
    print(numfails*100/533)
    falseneg = numfails
    print("false pos: " + str(falsepos) + " falseneg: " + str(falseneg))
pred()
