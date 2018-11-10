# Denoising Autoencoder for MNIST dataset

This project is a convolution based autoencoder written in Python using Tensorflow that takes noisy images from MNIST dataset and performs noise removal to return denoised images. It uses high levelTensorflow API for designing, training, and testing of the network.
****
## Getting Started
Clone the repository or simply download it.

### Prerequisites
The project requires you to have python3, tensorflow, numpy, matplotlib(for visualizing results) and an internet connection(to download the dataset for the first time). 

### Requirements
1. Python 3

2. Tensorflow

3. numpy

4. matplotlib

## Deployment
1. To train the model, use ``python autoencoder_train.py``

2. If you want to use the pretrained model, uncomment the 'load previously trained model' section and comment the 'training params', 'input layer', 'encoding', 'decoding', 'output layer', 'model' and 'train' sections.

3. For prediction using the trained model, do ``python autoencoder_predict.py``

## Results

Achieved a test accuracy of 81%.

Prediction results on test images from MNIST. Row 1 has noisy images. Row 2 are the denoised images given by the autoencoder.
![Screenshot](Figure.png)

## Built With

* [Python](https://www.python.org)
* [Tensorflow](https://www.tensorflow.org)
* [NumPy](http://www.numpy.org/)
* [Matplotlib](https://matplotlib.org/)
