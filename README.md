Applied Machine Learning Models

This repository contains practical implementations of essential Machine Learning algorithms built using Python and scikit-learn.

The primary goal is to strengthen conceptual clarity through implementation. Each notebook focuses on applying algorithms to structured datasets, analyzing performance, and understanding how model choices affect results.

This repository emphasizes:

Real-world experimentation

Systematic feature selection

Model tuning and validation

Performance comparison

Proper evaluation techniques

📁 Project Structure

1️⃣ auto_mpg_backward_elimination.ipynb

Implements Linear Regression using a backward elimination strategy to identify the most relevant predictors.

What it demonstrates:

Removing features step-by-step based on statistical relevance

Validating model performance using cross-validation

Evaluating regression quality using R² score

Interpreting feature contribution

2️⃣ auto_mpg_forward_feature_selection.ipynb

Applies forward selection for building an optimized regression model.

What it demonstrates:

Incrementally adding features

Measuring improvement after each addition

Selecting the most impactful combination of variables

3️⃣ svm_grid_search.ipynb

Builds and tunes a Support Vector Machine classifier.

What it demonstrates:

Testing multiple kernel functions

Applying Grid Search for parameter optimization

Selecting the best model using cross-validation

Evaluating generalization performance

4️⃣ confusion_matrix_analysis.ipynb

Explores classification model evaluation techniques.

What it demonstrates:

Constructing and interpreting confusion matrices

Calculating Accuracy, Precision, Recall, and F1-score

Understanding different types of classification errors

Comparing model performance using multiple metrics

5️⃣ knn_breastcancer_implementation.ipynb

Implements K-Nearest Neighbors (KNN) for tumor classification using the Breast Cancer dataset.

What it demonstrates:

Loading and preparing the dataset

Splitting into training and testing sets

Training models with different values of K (1–7)

Comparing performance across multiple K values

Evaluating predictions using accuracy and confusion matrix

This notebook highlights how distance-based algorithms behave when the number of neighbors changes.

6️⃣ knn_categorical_job_prediction.ipynb

Applies KNN to a fully categorical dataset for predicting job selection outcomes.

Objective:
Determine whether a candidate receives a job offer (Yes/No) based on academic and skill-related attributes.

Attributes Used:

CGPA

Communication Skills

Aptitude

Professional Skills

Implementation Steps:

Import dataset from CSV

Convert target labels into binary format

Transform categorical features using One-Hot Encoding

Split data into training and testing subsets

Train KNN models with multiple K values (1–5)

Evaluate model performance using confusion matrix and accuracy

This notebook demonstrates how categorical data must be converted into numerical form before applying distance-based algorithms.

📌 Topics Covered

Linear Regression

K-Nearest Neighbors (KNN)

Feature Selection Methods

Cross-Validation

Hyperparameter Tuning

Support Vector Machines (SVM)

Classification Metrics

Confusion Matrix Evaluation

🧰 Tools & Libraries

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn
