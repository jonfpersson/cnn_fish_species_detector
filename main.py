#importing other required libraries
from vgg19 import train
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Prepare datset
datagen = ImageDataGenerator()
#load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/', class_mode='categorical', batch_size=30, target_size=(224, 224))
# load and iterate validation dataset
val_it = datagen.flow_from_directory('data/validation/', class_mode='categorical', batch_size=30, target_size=(224, 224))

#Callback
ck = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
train(ck, train_it, val_it)