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

## **4 . 🔑 Key Features:**
- 📩 **Email Classification**: Predicts whether an email is spam or not.
- 📊 **CountVectorizer** for text feature extraction.
- 🎯 **High accuracy** of 97.13%.
- 🖼️ Confusion matrix visualization to evaluate model performance.

. Data Preprocessing

The following preprocessing steps were applied to the email text:

Conversion to lowercase

Removal of punctuation and special characters

Removal of stop words

Tokenization

Feature extraction using TF-IDF Vectorization (for machine learning models)

For the deep learning model, text was converted into sequences using tokenization and padding.
