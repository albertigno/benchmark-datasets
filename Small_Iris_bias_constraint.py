import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.constraints import max_norm
import os
import h5py

dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, 'iris.csv')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = read_csv(filepath, header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

nsamples = X.shape[0]
m = X.shape[1]
n = 1

X = X.astype('float32')

X = normalize(X, norm='max', axis=0)
#Y = Y.astype('float32')
#X = swapaxes(swapaxes(X, 2, 3), 1, 2)
#X = X.reshape(nsamples, 1, m, n)
#X /= 255.0



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=10)
Y_test = np_utils.to_categorical(Y_test, 3)

numpy.savez('xtest_iris',X_test)
numpy.savez('ytest_iris',Y_test)

# define baseline model
def baseline_model():
  # create model
  model = Sequential()

  model.add(Dense(20, input_dim=4, kernel_initializer= 'glorot_uniform' , kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation= 'relu' ))
  model.add(Dense(10, kernel_initializer='glorot_uniform', kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation='relu'))
  model.add(Dense(3, kernel_initializer= 'glorot_uniform' , kernel_constraint=max_norm(1.), bias_constraint=max_norm(0.4), activation= 'softmax' ))
  # Compile model
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  return model

model = baseline_model()
# Fit the model
model.fit(X, dummy_y, validation_data=(X_test, Y_test), epochs=1000, batch_size=1,
    verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Error: %.2f%%" % (100-scores[1]*100))



model.save("iris_bias_constraint.h5")
print("Saved model to disk")