import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def add_noise(data, noise = 0.2):
    data = data + np.random.randn(*data.shape) * noise
    data = data.clip(0., 1.)
    return data

def preprocess_data(data):
    data = data.reshape(-1, 28, 28, 1)
    data = data/255.0
    return data

#params
num_images = 10

#dataset
mnist = tf.keras.datasets.mnist
_, test_data = mnist.load_data()
x_test = test_data[0]
x_test = preprocess_data(x_test[0:num_images])
x_test = add_noise(x_test)

#prediction
autoencoder = models.load_model('./model/autoencoder.h5')
out = autoencoder.predict(x_test.reshape(num_images, 28, 28, 1))

#results
plt.figure()
for i in range(0, num_images):
    plt.subplot(2, num_images, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.subplot(2, num_images, num_images + i+1)
    plt.imshow(out[i].reshape(28,28), cmap='gray')
plt.show()