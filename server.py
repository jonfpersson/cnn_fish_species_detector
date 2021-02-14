#!/usr/bin/env python
import socket
from threading import Thread
import numpy as np
import os
import argparse
#from sklearn.externals import joblib
import joblib
import traceback
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from keras.preprocessing import image

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = load_model('vgg19Model.h5')
print('Model loaded')
server_address = ('0.0.0.0', 4224)
buffer_size = 4096


print('Warming up the model')
start = time.perf_counter()
input_shape = (1, ) + (224, 224, 3)
dummpy_img = np.ones(input_shape)
dummpy_img = preprocess_input(dummpy_img)
model.predict(dummpy_img)
end = time.perf_counter()
print('Warming up took {} s'.format(end - start))

FILE_DOES_NOT_EXIST = '-1'
UNKNOWN_ERROR = '-2'
classes = ['Brax', 'Debris', 'Lying' ,'Perch', 'Pike' , 'Roach']


def handle(clientsocket):
    while 1:
        buf = clientsocket.recv(buffer_size)
        if buf == 'exit'.encode():
            return  # client terminated connection

        response = ''
        if os.path.isfile(buf):
            try:
                img = buf.decode()

                img = cv2.imread(img)
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
                print("Response")
                print(response)
            except Exception as e:
                print('Error', e)
                traceback.print_stack()
                response = UNKNOWN_ERROR
        else:
            response = FILE_DOES_NOT_EXIST

        clientsocket.sendall(response.encode())


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversocket.bind(server_address)
serversocket.listen(10000)

print('Ready for requests')

while 1:
    # accept connections from outside
    (clientsocket, address) = serversocket.accept()

    ct = Thread(target=handle, args=(clientsocket,))
    ct.run()
