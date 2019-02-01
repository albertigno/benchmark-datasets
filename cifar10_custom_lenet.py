# -*- coding: utf-8 -*-

# Simple CNN for the CIFAR10 Dataset
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define a simple CNN model

batch_size    = 16
epochs        = 200
iterations    = 391
num_classes   = 10
weight_decay  = 0.0001
mean          = [125.307, 122.95, 113.865]
std           = [62.9932, 62.0887, 66.7048]

def lenet_model():
    model = Sequential()
    model.add(Conv2D(96, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), input_shape=(3,32,32)))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(192, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4) ))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4) ))
    model.add(Dense(1024, activation = 'relu', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4) ))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4) ))
    optimizer = SGD(lr=0.0001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# modelo 2: alexnet
def alexnet_model():
    model = Sequential()
    model.add(Conv2D(32, 32, 32, border_mode='same', input_shape=(3, 32, 32), kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2,2), strides=2))

    model.add(Conv2D(64, 16, 16, border_mode='same', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2,2), strides=2))

    model.add(Conv2D(128, 8, 8, border_mode='same', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(128, 8, 8, border_mode='same', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(128, 4, 4, border_mode='same', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2,2), strides=2))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), kernel_initializer='he_normal'))
    #model.add(Dense(256, activation='relu', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), kernel_initializer='he_normal'))

    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4) ))
    optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def simple_model():
    # create model
    model = Sequential()
    model.add(Conv2D(5, (5, 5), input_shape=(3, 32, 32), kernel_initializer= 'glorot_uniform' , kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(20,  kernel_initializer= 'glorot_uniform', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
#model = lenet_model()
model = alexnet_model()
# using real-time data augmentation
#print('Using real-time data augmentation.')
#datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20,
#        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
#
#datagen.fit(X_train)
#
## start train 
#model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),
#                    steps_per_epoch=iterations,
#                    epochs=epochs,
#                    validation_data=(X_test, y_test))

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=batch_size,
    verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save("./models/alexnet_cifar10_constraint.h5")
print("Saved model to disk")# -*- coding: utf-8 -*-

