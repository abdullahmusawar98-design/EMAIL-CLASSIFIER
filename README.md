# 📧 Spam Email Classifier
## 1. Introduction

Email spam detection is an important application of machine learning and natural language processing. Spam emails can contain advertisements, misleading content, or malicious links. Manual filtering is inefficient, therefore automated classification systems are required to identify spam emails accurately.

This project focuses on classifying emails as Spam or Ham (Not Spam) using both traditional machine learning models and deep learning techniques.

## 2. Problem Statement

Email spam has become a major challenge in digital communication systems. Spam emails not only waste users’ time but also pose serious security threats such as phishing and malware attacks. Manual filtering techniques are inefficient and unreliable. Therefore, there is a strong need for an automated email classification system that can accurately distinguish between spam and legitimate emails using machine learning and deep learning techniques.

## 3. Objectives


The objectives of this project are:

To build an automated email spam classification system

To compare traditional machine learning models with deep learning models

To evaluate model performance using Confusion Matrix and ROC–AUC score

## 4. Dataset Description

The dataset used in this project contains labeled email messages.

https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv

## 5 . 🔑 Key Features
- 📩 **Email Classification**: Predicts whether an email is spam or not.
- 📊 **CountVectorizer** for text feature extraction.
- 🎯 **High accuracy** of 97.13%.
- 🖼️ Confusion matrix visualization to evaluate model performance.

## 6. Data Preprocessing

The following preprocessing steps were applied to the email text:

Conversion to lowercase

Removal of punctuation and special characters

Removal of stop words

Tokenization

Feature extraction using TF-IDF Vectorization (for machine learning models)

For the deep learning model, text was converted into sequences using tokenization and padding.

## 7. Models Implemented
### **7.1 Linear Regression Model:**

Linear Regression was implemented as a baseline approach. Since it produces continuous outputs, a threshold value was applied to classify emails into spam or ham.

Key characteristics:

Simple baseline model

Not naturally designed for classification tasks

### **7.2 Logistic Regression Model:**

Logistic Regression was used as a classification model that estimates the probability of an email being spam.

**Key characteristics:**

Designed for binary classification

Performs well with high-dimensional text data

Produces interpretable probability scores

### **7.3 Deep Learning Model (Neural Network):**

A deep learning-based neural network was implemented to capture complex patterns in email text.

Model architecture (example):

Embedding layer

One or more hidden dense layers

Output layer with sigmoid activation

**Advantages:**

Learns non-linear relationships

Captures semantic patterns in text

## 8. Model Evaluation
### **8.1 Confusion Matrix:**

The confusion matrix was used to analyze the classification results of each model.

**Definitions:**

**True Positives (TP):** Spam correctly identified

**True Negatives (TN):** Ham correctly identified

**False Positives (FP):** Ham incorrectly identified as spam

**False Negatives (FN):** Spam incorrectly identified as ham

### **Linear Regression – Confusion Matrix:**

TP =  595
TN =  140
FP =  86
FN =  214

### **Logistic Regression – Confusion Matrix:**

TP =  716
TN =  19
FP =  10
FN =  290

### **Neural Network – Confusion Matrix:**

TP =  538
TN =  13
FP =  10
FN =  215

### 8.3 ROC–AUC Score
The ROC–AUC score measures the ability of a model to distinguish between spam and ham emails.

### **Linear Regression ROC–AUC:**
ROC–AUC Score = 
Interpretation:

### **Logistic Regression ROC–AUC**
ROC–AUC Score = 
Interpretation:

### **Neural Network ROC–AUC**
ROC–AUC Score = 
Interpretation:

### 8.4 Accuracy


## 9. Results 
### 9.1 Linear Regression



### 9.2 Logistic Regression



### 9.3 Deep Learning(Neural Network)



## 10. Future Enhancements

-  **TF-IDF**: Implementing TF-IDF for feature extraction instead of simple word counts.
-  **Additional Classifiers**: Try SVM or Random Forest for performance comparison.
-  **Real-time Classification**: Integrate with email clients for real-time spam detection.
-  **Phishing Detection**: Enhance to detect phishing emails using additional features like email headers.


