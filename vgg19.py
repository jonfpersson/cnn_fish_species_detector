
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def train(ck, train_it, val_it):
    #Definition of hyperparameters and optimizer
    learn_rate=.0001
    opt= SGD(lr=learn_rate, momentum=.9, nesterov=False)

    base_model_VGG19 = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=3)
    #base_model_VGG19.summary()

    #Adding the final layers to the above base model where the actual classification is done in the dense layers
    model_vgg19 = Sequential()
    model_vgg19.add(base_model_VGG19) 
    model_vgg19.add(Flatten()) 
    model_vgg19.add(Dense(1024,activation=('relu'),input_dim=512))
    model_vgg19.add(Dense(512,activation=('relu')))
    model_vgg19.add(Dense(256,activation=('relu'))) 
    model_vgg19.add(Dropout(.3))
    model_vgg19.add(Dense(128,activation=('relu')))
    #model_vgg19.add(Dense(64,activation=('relu')))
    model_vgg19.add(Dropout(.2))
    model_vgg19.add(Dense(5,activation=('softmax')))
    
    #Compiling and training the VGG19 model
    model_vgg19.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_vgg19.fit(train_it,
                    epochs=150,
                    batch_size=30,
                    steps_per_epoch = 30,
                    validation_data = val_it,
                    validation_steps = 250,
                    callbacks = ck,
                    verbose = 1)

    print('Saving model')
    model_vgg19.save("vgg19Model.h5")