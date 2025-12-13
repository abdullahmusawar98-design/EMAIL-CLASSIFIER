# 📧 Spam Email Classifier
# 1. Introduction

Email spam detection is an important application of machine learning and natural language processing. Spam emails can contain advertisements, misleading content, or malicious links. Manual filtering is inefficient, therefore automated classification systems are required to identify spam emails accurately.

This project focuses on classifying emails as Spam or Ham (Not Spam) using both traditional machine learning models and deep learning techniques.

# 2. Objectives

The objectives of this project are:

To build an automated email spam classification system

To compare traditional machine learning models with deep learning models

To evaluate model performance using Confusion Matrix and ROC–AUC score

# 3. Dataset Description

The dataset used in this project contains labeled email messages.

https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

# 4 . 🔑 Key Features
- 📩 **Email Classification**: Predicts whether an email is spam or not.
- 📊 **CountVectorizer** for text feature extraction.
- 🎯 **High accuracy** of 97.13%.
- 🖼️ Confusion matrix visualization to evaluate model performance.

# 5. Data Preprocessing

The following preprocessing steps were applied to the email text:

Conversion to lowercase

Removal of punctuation and special characters

Removal of stop words

Tokenization

Feature extraction using TF-IDF Vectorization (for machine learning models)

For the deep learning model, text was converted into sequences using tokenization and padding.

# 5. Models Implemented
## **5.2 Linear Regression Model:**

Linear Regression was implemented as a baseline approach. Since it produces continuous outputs, a threshold value was applied to classify emails into spam or ham.

Key characteristics:

Simple baseline model

Not naturally designed for classification tasks

## **5.2 Logistic Regression Model:**

Logistic Regression was used as a classification model that estimates the probability of an email being spam.

Key characteristics:

Designed for binary classification

Performs well with high-dimensional text data

Produces interpretable probability scores

## **5.3 Deep Learning Model (Neural Network):**

A deep learning-based neural network was implemented to capture complex patterns in email text.

Model architecture (example):

Embedding layer

One or more hidden dense layers

Output layer with sigmoid activation

Advantages:

Learns non-linear relationships

Captures semantic patterns in text

# 6. Model Evaluation
## **6.1 Confusion Matrix:**

The confusion matrix was used to analyze the classification results of each model.

Definitions:

True Positives (TP): Spam correctly identified

True Negatives (TN): Ham correctly identified

False Positives (FP): Ham incorrectly identified as spam

False Negatives (FN): Spam incorrectly identified as ham

## **Linear Regression – Confusion Matrix:**

TP =  
TN =  
FP =  
FN =  

## **Logistic Regression – Confusion Matrix:**

TP =  
TN =  
FP =    
FN =  

## **Neural Network – Confusion Matrix:**

TP =  
TN =  
FP =  
FN =  

# 7. ROC–AUC Score
The ROC–AUC score measures the ability of a model to distinguish between spam and ham emails.

## **Linear Regression ROC–AUC:**
ROC–AUC Score = 
Interpretation:

## **Logistic Regression ROC–AUC**
ROC–AUC Score = 
Interpretation:

## **Neural Network ROC–AUC**
ROC–AUC Score = 
Interpretation:

# 8. Future Enhancements
## **🌱 Future Enhancements**

- 🌐 **TF-IDF**: Implementing TF-IDF for feature extraction instead of simple word counts.
- ⚙️ **Additional Classifiers**: Try SVM or Random Forest for performance comparison.
- 💡 **Real-time Classification**: Integrate with email clients for real-time spam detection.
- 🛡️ **Phishing Detection**: Enhance to detect phishing emails using additional features like email headers.


