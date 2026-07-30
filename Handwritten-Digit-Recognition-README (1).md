# Handwritten Digit Recognition using CNN

## 📌 Project Overview

This project focuses on developing a Deep Learning model for recognizing
handwritten digits (0-9) using a Convolutional Neural Network (CNN).

The model is trained on the MNIST dataset, which contains handwritten
digit images. The goal is to classify input images and predict the
correct digit with high accuracy.

This project demonstrates the application of Deep Learning and Computer
Vision techniques in image classification tasks.

## 🎯 Project Objectives

-   Build a Convolutional Neural Network (CNN) model for handwritten
    digit classification.
-   Apply image preprocessing techniques to prepare data for training.
-   Train and evaluate the deep learning model using TensorFlow and
    Keras.
-   Analyze model performance using accuracy and loss metrics.
-   Test the model's ability to recognize unseen handwritten digits.

## 🛠️ Technologies Used

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   Matplotlib
-   Google Colab
-   Convolutional Neural Network (CNN)

## 📂 Dataset

The project uses the MNIST dataset.

Dataset details: - Total images: 70,000 grayscale images - Training
images: 60,000 - Testing images: 10,000 - Image size: 28 × 28 pixels -
Classes: Digits from 0 to 9

## ⚙️ Data Preprocessing

Applied preprocessing steps: - Loading the MNIST dataset. - Normalizing
pixel values. - Reshaping images for CNN input. - Preparing training and
testing data.

## 🧠 CNN Model Architecture

The model consists of:

-   Conv2D layers for feature extraction.
-   MaxPooling layers for dimensionality reduction.
-   Flatten layer to convert features into vectors.
-   Dense layers for classification.
-   Softmax output layer for predicting 10 digit classes.

## 🚀 Model Training

Training configuration: - Optimizer: Adam - Loss Function: Sparse
Categorical Crossentropy - Metric: Accuracy - Epochs: 5

## 📊 Results

The CNN model achieved more than 99% accuracy on the MNIST test dataset.

The performance was evaluated using: - Training Accuracy - Validation
Accuracy - Training Loss - Validation Loss - Prediction Analysis

## 📁 Project Structure

    Handwritten-Digit-Recognition-CNN/

    │── handwritten_digit_recognition.py
    │── digit_recognition_model.h5
    │── README.md
    │
    └── screenshots/
        ├── accuracy_loss.png
        └── predictions.png

## ▶️ How to Run

Install required libraries:

    pip install tensorflow numpy matplotlib

Run the project:

    python handwritten_digit_recognition.py

## 🔮 Future Improvements

-   Apply data augmentation techniques.
-   Add dropout layers to improve generalization.
-   Test advanced CNN architectures.
-   Deploy the model as a web or mobile application.
-   Support additional handwriting datasets.

## 📚 References

-   TensorFlow Documentation
-   MNIST Dataset
-   Deep Learning with Python
