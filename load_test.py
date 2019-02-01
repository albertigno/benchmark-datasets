# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model

# para leer imagenes y graficar
from scipy import misc
import matplotlib.pyplot as plt

# para text to speech
from gtts import gTTS
import os
import time

# leer imagenes
image1 = misc.imread('uno.png', mode='L')
image2 = misc.imread('dos.png', mode='L')
image3 = misc.imread('tres.png', mode='L')
image4 = misc.imread('cuatro.png', mode='L')

# crear ndarray con imagenes propias
own_images = numpy.zeros((4, 1,28,28))
own_images[0] = image1
own_images[1] = image2
own_images[2] = image3
own_images[3] = image4

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

test_images = X_test[:4]

model = load_model('large_mnist.h5')

predictions1 = numpy.argmax(model.predict(test_images),1)
predictions2 = numpy.argmax(model.predict(own_images),1)

print('predicciones MNIST: '+ str(predictions1))
print('predicciones propias: '+ str(predictions2))

tts = gTTS(text='Los numeros son', lang='es')
tts.save("sound.mp3")
os.system("mpg321 sound.mp3")

for k in range(4):
    text = str(predictions2[k])
    tts = gTTS(text=text, lang='es')
    tts.save("sound.mp3")
    os.system("mpg321 sound.mp3")
    time.sleep(0.5)

plt.figure('MNIST')
plt.subplot(141), plt.imshow(test_images[0][0])
plt.subplot(142), plt.imshow(test_images[1][0])
plt.subplot(143), plt.imshow(test_images[2][0])
plt.subplot(144), plt.imshow(test_images[3][0])

plt.figure('propias')
plt.subplot(141), plt.imshow(own_images[0][0])
plt.subplot(142), plt.imshow(own_images[1][0])
plt.subplot(143), plt.imshow(own_images[2][0])
plt.subplot(144), plt.imshow(own_images[3][0])
plt.show()
