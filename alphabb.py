# ### 1. import data

from mnist import MNIST
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# data = MNIST(path='data/', return_type='numpy')
# data.select_emnist('letters')
# X, y = data.load_training()

# X.shape, y.shape
# X = X.reshape(124800, 28, 28)
# y = y.reshape(124800, 1)

# # list(y) --> y ranges from 1 to 26

# y = y-1

# # list(y) --> y ranges from 0 to 25 now


# ### 2. train-test split

# # pip install scikit-learn
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

# # (0,255) --> (0,1)
# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255

# # y_train, y_test

# # pip install tensorflow
# # integer into one hot vector (binary class matrix)
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train, num_classes = 26)
# y_test = np_utils.to_categorical(y_test, num_classes = 26)

# #y_train, y_test


# ### 3. Define our model

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten

# model = Sequential()
# # model.add(Flatten(input_shape = (28,28)))
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.2)) # preventing overfitting
# # model.add(Dense(512, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(26, activation='softmax'))
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


# model.summary()

# model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])



# ### 4. calculate accuracy

# # before training
# score = model.evaluate(X_test, y_test, verbose=0)
# accuracy = 100*score[1]
# print("Before training, test accuracy is", accuracy)

# # let's train our model
from keras.callbacks import ModelCheckpoint

# checkpointer = ModelCheckpoint(filepath = 'best_model.h5', verbose=1, save_best_only = True)
# model.fit(X_train, y_train, batch_size = 128, epochs= 10, validation_split = 0.2, 
#           callbacks=[checkpointer], verbose=1, shuffle=True)

# model.load_weights('best_model.h5')

# # calculate test accuracy
# score = model.evaluate(X_test, y_test, verbose=0)
# accuracy = 100*score[1]

# print("Test accuracy is ", accuracy)










# Import necessary libraries
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
data = MNIST(path='data/', return_type='numpy')
data.select_emnist('digits')
X, y = data.load_training()

X.shape, y.shape
X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)

# list(y) --> y ranges from 1 to 26

y = y-1

# list(y) --> y ranges from 0 to 25 now


### 2. train-test split

# pip install scikit-learn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)


# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
# model.fit(X_train, y_train,
#           batch_size=128,
#           epochs=10,
#           verbose=1,
#           validation_data=(X_test, y_test))

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)

checkpointer = ModelCheckpoint(filepath = 'best_model.h5', verbose=1, save_best_only = True)
model.fit(X_train, y_train, batch_size = 128, epochs= 3, validation_split = 0.2, 
          callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights('best_model.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions
predictions = model.predict(X_test)
print(np.argmax(predictions[0])) # Print the predicted number for the first test image

# Plot the first test image
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.show()