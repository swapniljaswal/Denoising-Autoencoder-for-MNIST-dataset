import tensorflow as tf
from tensorflow.keras import layers, Model, models
import numpy as np

def add_noise(data, mu = 0.5, sigma = 0.3):
    noise = np.random.normal(mu, sigma, data.shape)
    data = data + noise
    data = data/np.max(data)
    return data

def preprocess_data(data):
    data = data.reshape(-1, 28, 28, 1)
    data = data/255.0
    return data

#dataset
mnist = tf.keras.datasets.mnist
train_data, test_data = mnist.load_data()
x_train = train_data[0]
x_test = test_data[0]

#preprocessing data
x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)
x_train_noisy = add_noise(x_train)
x_test_noisy  = add_noise(x_test)

'''
#load previously trained model
autoencoder = models.load_model('./model/autoencoder.h5')
'''

#training params
epochs = 200
batch_size = 128
val_split = 0.3

#input layer
inputs = layers.Input(shape = (28, 28, 1))

#encoding
conv = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
pool = layers.MaxPool2D(padding='same')(conv)
conv = layers.Conv2D(8, 3, activation='relu', padding='same')(pool)
pool = layers.MaxPool2D(padding='same')(conv)
conv = layers.Conv2D(8, 3, activation='relu', padding='same')(pool)
encoded = layers.MaxPooling2D(padding='same')(conv)

#decoding
conv = layers.Conv2D(8, 3, activation= 'relu', padding= 'same')(encoded)
up = layers.UpSampling2D()(conv)
conv = layers.Conv2D(8, 3, activation= 'relu', padding= 'same')(up)
up = layers.UpSampling2D()(conv)
conv = layers.Conv2D(16, 3, activation= 'relu')(up)
up = layers.UpSampling2D()(conv)

#output layer
outputs = layers.Conv2D(1, 3, activation='sigmoid', padding = 'same')(up)

#model
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

#train
autoencoder.fit(x_train_noisy, x_train, epochs = epochs, batch_size = batch_size, validation_split = val_split, verbose = 1)
autoencoder.save('./model/autoencoder.h5')

#test
evaluation = autoencoder.evaluate(x_test_noisy, x_test)
evaluation = np.around(evaluation, 2)
print('Test Accuracy:', evaluation[1]*100, '%, Test Loss:', evaluation[0])

