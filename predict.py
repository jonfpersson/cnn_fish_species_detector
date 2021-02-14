import  tensorflow as tf
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def evaluateModel():
    datagen = ImageDataGenerator()
    test_it = datagen.flow_from_directory('data/test', class_mode='categorical', batch_size=30, target_size=(224, 224))
    loss = model.evaluate(test_it, steps=100, verbose = 1)
    print(f'Test loss: {loss[0]} / Test accuracy: {loss[1]}')

def get_prob():
    #Load image and reshape
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)

    out = model.predict(np.array(img))
    prediction = np.argmax(out)
    top10 = out[0].argsort()[-10:][::-1]

    class_indices = dict(zip(classes, range(len(classes))))
    keys = list(class_indices.keys())
    values = list(class_indices.values())

    predicted_class = keys[values.index(prediction)]
    propability = out[0][values.index(prediction)]
    response = '{"class":"%s", "probability":"%s"}' % (predicted_class, propability)
    print(response)

#Configure memory settings
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
 

# load and iterate test dataset
model = load_model('vgg19Model.h5')

#Add argument to script
parser = argparse.ArgumentParser()
parser.add_argument('--path')
args = parser.parse_args()
path = args.path

classes = ['Brax', 'Debris','Perch', 'Pike' , 'Roach']

#get_prob()
evaluateModel()     
