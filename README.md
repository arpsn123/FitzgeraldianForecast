# Text Prediction Using LSTM on "The Great Gatsby"

<!-- Repository Overview Badges -->
<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<!-- Issue & Pull Request Badges -->
<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<!-- Repository Activity & Stats Badges -->
<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<!-- Language & Code Style Badges -->
<div align="center">
    <img src="https://img.shields.io/github/languages/count/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count">
    <img src="https://img.shields.io/github/languages/top/arpsn123/FitzgeraldianForecast?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<!-- Maintenance Status Badge -->
<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>


This project implements a Long Short-Term Memory (LSTM) model to predict the next character in a sequence based on the text from "The Great Gatsby" by Francis Scott Key Fitzgerald. The model captures the unique tone and style of Fitzgerald's writing, showcasing the power of deep learning in natural language processing. By utilizing advanced machine learning techniques, this project demonstrates how artificial intelligence can mimic literary styles, providing insights into both technology and literature.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Output](#model-output)
- [Training](#training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

## Introduction

The LSTM model is specifically designed for sequence prediction tasks and is well-suited for generating text due to its ability to learn long-term dependencies. In this project, we focus on character-level prediction, where the model is trained on the entire text of "The Great Gatsby." This allows it to generate coherent sequences that reflect the stylistic and thematic elements of Fitzgerald's work.

## Tech Stack



![Python](https://img.shields.io/badge/python-3.x-blue) **Python**: The primary programming language used for developing the LSTM model.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) **TensorFlow**: An open-source library for building and deploying machine learning models.

![Keras](https://img.shields.io/badge/Keras-2.x-brightgreen) **Keras**: A high-level API for fast and easy model building in TensorFlow.

![NumPy](https://img.shields.io/badge/NumPy-1.21.2-red) **NumPy**: A fundamental library for numerical computing and data manipulation.

![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-purple) **Matplotlib**: A plotting library for creating static and interactive visualizations.

![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-6.x-yellow) **Jupyter Notebook**: A web application for creating and sharing documents with live code and visualizations.

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.x-green) **Scikit-learn**: A library for machine learning offering various algorithms for classification and regression.


## Data Preprocessing

Before training, the text data undergoes significant preprocessing:
- **Lowercasing**: All text is converted to lowercase to ensure uniformity.
- **Character Removal**: Numbers and special characters are removed, leaving only alphabetic characters and spaces.
- **Tokenization**: The text is split into individual characters, and a mapping of characters to integers is created. This mapping is essential for converting text into a format suitable for training the model.

## Requirements

To successfully run this project, ensure that you have the following software installed:
- **Python 3.x**: The programming language used for this project, known for its simplicity and flexibility in data science and machine learning applications.
- **TensorFlow/Keras**: A powerful library for building and training machine learning models. Keras, a high-level API for TensorFlow, simplifies the model-building process.
- **NumPy**: A fundamental package for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

## Installation

Follow these steps to set up the project on your local machine:
1. Clone the repository:
   - Command: `git clone https://github.com/arpsn123/text-prediction-lstm.git`
   - Navigate into the directory: `cd text-prediction-lstm`
   
2. Install the required packages:
   - Command: `pip install -r requirements.txt`
   - This command will install all the necessary dependencies listed in the requirements.txt file, ensuring that your environment is set up correctly.

## Usage

To run the model training, execute the following command in your terminal:
- Command: `python train_model.py`
- This command will initiate the training process using the preprocessed text data from "The Great Gatsby." During training, the model will learn to predict the next character based on the preceding characters in the input sequence.

## Model Output

After training, the model will save the trained model weights and architecture as `the_great_gatsby_10+50_epochs.hdf5`. This file can be reused for further evaluation or for generating text sequences.

## Training

The model is built using the following architecture:
- **LSTM Layers**: Three stacked LSTM layers designed to capture complex temporal dependencies within the text data.
- **Dense Layers**: Two Dense layers, with the final layer employing softmax activation to facilitate multi-class character prediction.

### Training Process
- **Input Sequence Length**: 60 characters — the model looks at 60 characters of input to predict the next character.
- **Batch Size**: 128 — the number of training samples processed before the model is updated.
- **Epochs**: Initially trained for 10 epochs, followed by an additional 50 epochs.

The model uses categorical cross-entropy as the loss function and the RMSprop optimizer, ensuring efficient training and convergence.

## Model Evaluation

After training, the model can be evaluated on the training data. To load and evaluate the model, use the following command in a Python environment:
- Command: `from keras.models import load_model`
- Load the model: `model = load_model('the_great_gatsby_10+50_epochs.hdf5', compile=False)`
- Compile the model: `model.compile('adam', loss='categorical_crossentropy')`

This will load the trained model and compile it for evaluation.

## Results

The model achieved a loss of approximately 2.2992 after an additional 50 epochs of training, showing improvements in character prediction accuracy. This indicates that the model effectively learned the intricate details of Fitzgerald's writing style.


