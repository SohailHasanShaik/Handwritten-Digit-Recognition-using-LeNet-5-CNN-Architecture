# Handwritten-Digit-Recognition-using-LeNet-5-CNN-Architecture on MNIST Database

## Overview

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The MNIST dataset is a well-known dataset in the machine learning community, consisting of 70,000 grayscale images of handwritten digits (0-9), with each image being 28x28 pixels in size.

The project is implemented using TensorFlow and Keras, and includes data preprocessing, model building, training, evaluation, and prediction.

## Dataset

The dataset used in this project is the MNIST dataset, which is available through TensorFlow's `datasets` module.

- **Number of Classes:** 10 (digits 0-9)
- **Number of Training Samples:** 60,000
- **Number of Testing Samples:** 10,000
- **Image Size:** 28x28 pixels, grayscale

## Project Structure

- `mnist_classification.py`: Main script containing the implementation of the model, training, and evaluation.
- `README.md`: Detailed documentation of the project.

### Required Libraries

- TensorFlow
- Keras
- NumPy
- Pandas
- scikit-learn

## Data Preprocessing

1. Loading the Data
2. Reshaping the Data
3. Normalizing the Data
4. One-Hot Encoding the Labels

## Model Architecture

The CNN model is built using Keras' Sequential API and consists of the following layers:

1. **Conv2D Layer:** 6 filters of size 5x5, activation function `tanh`.
2. **AveragePooling2D Layer:** Pooling size 2x2, stride 1x1.
3. **Conv2D Layer:** 16 filters of size 5x5, activation function `tanh`.
4. **AveragePooling2D Layer:** Pooling size 2x2, stride 2x2.
5. **Conv2D Layer:** 120 filters of size 5x5, activation function `tanh`.
6. **Flatten Layer:** Flatten the input.
7. **Dense Layer:** 84 units, activation function `tanh`.
8. **Dense Layer:** 10 units (number of classes), activation function `softmax`.

The model is compiled with the categorical cross-entropy loss function and Adam optimizer.

## Training the Model

The model is trained on the training data for 20 epochs with a batch size of 32. A validation set is used to monitor the model's performance during training.

## Model Evaluation

The model is evaluated on the test data, and the accuracy is reported.

## Results

The model achieves an accuracy of over 99% on the MNIST test set. The detailed results and metrics can be found in the training logs.

## Conclusion

This project demonstrates how to build, train, and evaluate a CNN for digit classification using the MNIST dataset. The model can be further improved by experimenting with different architectures, optimization techniques, and data augmentation.
