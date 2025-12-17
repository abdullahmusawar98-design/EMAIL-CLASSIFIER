# 📧 Spam Email Classifier
##  Introduction
##  Problem Statement
With the increasing number of unsolicited and irrelevant emails, spam email detection has become a critical problem in information security. This project aims to develop a classification model to identify whether an email is spam or legitimate (ham). We aim to use both classical machine learning approaches and deep learning to classify these emails based on their content.

##  Objectives
To build and evaluate models using Logistic Regression, Decision Trees, and Neural Networks.

To perform hyperparameter tuning to optimize model performance.

To compare models using standard evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

##  Dataset Description
The dataset used for this project is the Spam Email Message Collection Dataset. It is a collection of Email words that are labeled as either spam or ham. This dataset is widely used for text classification tasks, specifically for spam detection.

**Dataset Source:** 
The dataset is available IN Repository.

**Size:**

The dataset consists of 5,574 messages, each labeled as either spam or ham.

**Number of features:** There are two main columns in the dataset:

**class:** The label indicating whether the message is spam (1) or ham (0).

message: The content of the Email message, which needs to be classified.

**Features:**

**class (Target variable):**

**ham:** Legitimate messages (non-spam).

**spam:** Unsolicited messages (spam).

**message (Input feature):**
The actual Email Message content that needs to be classified as either spam or ham. This text data is unstructured and needs to be converted into a numerical format (via vectorization) to be used for model training.

##  Preprocessing
**Text Cleaning:**

Removed unnecessary characters, digits, and punctuation marks from the text messages.

Removed stopwords (common words like “the”, “is”, etc.) that do not contribute significantly to the meaning of the text.

Converted all text to lowercase to standardize the data.

**Text Vectorization:**

Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text messages into numerical features.

TF-IDF helps represent the importance of a word in a document relative to the entire corpus. This transformation converts the messages into a matrix of numerical values that can be used by machine learning models.

The max_features=5000 parameter limits the number of features to 5000 based on the highest importance words.

**Label Encoding:**

The labels (ham and spam) were encoded into numerical values (ham = 0, spam = 1) using LabelEncoder to make them compatible with machine learning algorithms.

##  Methodology
## Classical Machine Learning Approaches Used

**Logistic Regression:**

Logistic Regression is a statistical model commonly used for binary classification tasks. It predicts the probability of an instance belonging to a particular class (in this case, spam or ham).

**Hyperparameter Tuning:**

GridSearchCV was used to perform hyperparameter tuning for the Logistic Regression model. It explored different values for the regularization parameter (C) and solver options (liblinear and saga) to find the best-performing configuration.

**C:** The regularization parameter that controls the trade-off between fitting the model and penalizing large coefficients. A smaller value indicates stronger regularization, and a larger value indicates weaker regularization.

**solver:** Optimization algorithm used to fit the logistic regression model. liblinear is suitable for small datasets, and saga is more efficient for large datasets.

**Decision Tree Classifier:**

A Decision Tree is a tree-like structure used for classification tasks. It splits the data at each node based on the feature that best separates the data.

**Hyperparameter Tuning:**

The GridSearchCV method was used to tune several hyperparameters:

**max_depth:** The maximum depth of the tree. A deeper tree can capture more complexity, but it might overfit the data.

**min_samples_split:** The minimum number of samples required to split an internal node.

**min_samples_leaf:** The minimum number of samples required to be at a leaf node.

**criterion:** The function to measure the quality of a split. gini is the default method, but entropy can also be used for calculating splits based on information gain.

**Evaluation Metrics for Classical ML:**

**Accuracy:** Measures the percentage of correctly predicted instances.

**Precision:** Measures how many of the predicted positive instances were actually positive.

**Recall:** Measures how many of the actual positive instances were correctly identified.

**F1-Score:** The harmonic mean of precision and recall, used when there is an imbalance between the classes.

**Confusion Matrix:** A matrix showing the number of true positives, true negatives, false positives, and false negatives.

##  Deep Learning Approach Used

**Neural Network (Deep Learning):**

A Neural Network (also known as a Multilayer Perceptron) is a deep learning model that consists of multiple layers (input, hidden, and output) for learning complex patterns from the data.

**Architecture:**

**Input Layer:** The input layer has neurons corresponding to each feature in the dataset (5000 features after TF-IDF).

**Hidden Layers:** The network has two hidden layers with ReLU (Rectified Linear Unit) activation, which is commonly used in deep learning due to its non-linear nature and computational efficiency.

**Dropout Layers:** Dropout layers were added to prevent overfitting by randomly setting a fraction of the input units to 0 at each update during training time.

**Output Layer:** A single neuron with Sigmoid activation is used for binary classification, where the output represents the probability of the message being spam.

**Hyperparameter Tuning:**

**Optimization:** The Adam optimizer was used for training the neural network, as it adapts the learning rate based on the data.

**Loss Function:** Binary Cross-Entropy was used as the loss function for binary classification tasks (spam vs. ham).

**Early Stopping:** An EarlyStopping callback was used to stop the training if the validation loss does not improve for a certain number of epochs, which helps to prevent overfitting.

**Evaluation Metrics for Deep Learning:**

**Accuracy:** Measures how well the model classifies the messages into correct categories (spam or ham).

**Confusion Matrix:** Displays the counts of true positives, false positives, true negatives, and false negatives for the deep learning model.

**ROC Curve:** The Receiver Operating Characteristic (ROC) curve compares the true positive rate with the false positive rate at various classification thresholds. The AUC (Area Under Curve) metric indicates how well the model can distinguish between the two classes.

**Model Training:**
The neural network was trained for a maximum of 50 epochs with batch size = 32. The model used dropout layers for regularization, and the training stopped early if there was no improvement in the validation loss for 3 consecutive epochs.

##  Results & Analysis for SPAM

| Model              | Precision | Recall  | F1-Score  |   
|--------------------|-----------|----------|----------|
|Logistic Regression |   0.98    |   1.00   |   0.99   |
|Decision Tree       |   0.98    |   0.99   |   0.99   |                 
|Neural Network      |   0.99    |    0.99  |  0.99    |          

##  Results & Analysis for HAM

| Model              | Precision | Recall  | F1-Score  |     
|--------------------|-----------|----------|----------|
|Logistic Regression |   0.97    |  0.85    |    0.91  |          
|Decision Tree       |   0.93    |  0.85    |    0.89  |    
|Neural Network      |   0.94    |  0.91    |    0.93  |          



## Visualization of Results:

**Confusion Matrix:** Visualize the confusion matrix for each model to analyze false positives and false negatives.


**ROC Curves:** Compare the Receiver Operating Characteristic (ROC) curve and AUC for each model.


## Statistical Significance Tests

**t-tests:** We used t-tests to determine if the differences in accuracy between the models were statistically significant.

## Business Impact Analysis

**Spam Filtering Impact:** By improving the accuracy of spam detection, businesses can reduce the number of unwanted emails, saving time and resources.

**Model Comparison:** The Neural Network performed better than Logistic Regression and Decision Trees in terms of accuracy and AUC, making it the most suitable choice for real-world applications.

## Conclusion

The Neural Network model outperformed traditional models like Logistic Regression and Decision Trees in terms of accuracy, precision, recall, and AUC score.

While Logistic Regression and Decision Trees provided reasonable results, the Neural Network demonstrated the best potential for detecting spam emails.

## Future Work

Fine-tuning the Neural Network: Further tuning of the deep learning model could lead to even better performance.

Text Data Augmentation: Incorporating more data and feature engineering could improve model accuracy.

**Deployment:** Future steps involve deploying the model in a production environment where it can classify incoming emails in real-time.

## References
**https://github.com/SimarjotKaur/Email-Classifier**
**https://abhimishra91.github.io/prj_email/**
**https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/**
**https://www.geeksforgeeks.org/machine-learning/decision-tree/**
**https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/**

[Link 1](https://github.com/SimarjotKaur/Email-Classifier)<br>
[Link 2](https://abhimishra91.github.io/prj_email/)<br>
[Link 3](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)<br>
[Link 4](https://www.geeksforgeeks.org/machine-learning/decision-tree/)<br>
[Link 5](https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/)


