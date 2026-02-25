Applied Machine Learning Models

This repository contains structured implementations of core Machine Learning techniques using Python and scikit-learn.

The focus is on practical modeling, feature selection strategies, hyperparameter tuning, and performance evaluation using real datasets.

The goal of this repository is to strengthen conceptual understanding while following clean coding practices and reproducible experimentation workflows.

📂 Repository Structure

1️⃣auto_mpg_backward_elimination.ipynb

Implements Linear Regression with Backward Feature Elimination using cross-validation to identify the most impactful predictors.
Demonstrates model evaluation using R² score.

2️⃣ auto_mpg_forward_feature_selection.ipynb

Implements Forward Feature Selection to iteratively add significant features based on model performance improvement.

3️⃣ svm_grid_search.ipynb

Applies Support Vector Machine (SVM) with Grid Search for hyperparameter tuning across multiple kernels.
Demonstrates model optimization using cross-validation.

4️⃣ confusion_matrix_analysis.ipynb

Implements classification evaluation using confusion matrix and related performance metrics such as accuracy, precision, recall, and F1-score.

5️⃣ knn_breastcancer_implementation.ipynb

Implements K-Nearest Neighbors (KNN) classification on the Breast Cancer Wisconsin dataset.

This notebook demonstrates:

Loading built-in dataset using load_breast_cancer

Splitting data into training and testing sets

Training KNN models with different values of K (1 to 7)

Evaluating model performance using:

Confusion Matrix

Accuracy Score

Comparing accuracy across multiple neighbor values to identify optimal K

The implementation highlights how model performance changes as the number of neighbors varies and reinforces understanding of distance-based classification.

6️⃣ knn_categorical_job_prediction.ipynb

Implements K-Nearest Neighbors (KNN) classification on a small categorical dataset to predict whether a candidate receives a job offer.

This notebook demonstrates how to apply KNN on purely categorical data by converting it into numerical form before training the model.

📌 Problem Statement

Predict Job_offered (Yes/No) based on candidate attributes:

CGPA

Communication

Aptitude

Professional Skills

⚙️ Implementation Workflow

Load dataset from CSV

Convert target variable (Yes/No) into binary format (1/0)

Apply One-Hot Encoding to transform categorical features into numerical vectors

Split dataset into training and testing sets (70:30)

Train KNN models with multiple values of K (1–5)

Evaluate performance using:

Confusion Matrix

Accuracy Score


🔍 Key Concepts Covered

Linear Regression

K-Nearest Neighbors (KNN)

Feature Selection (Forward & Backward)

Cross-Validation

Hyperparameter Tuning

Support Vector Machines (SVM)

Model Evaluation Metrics

Confusion Matrix Analysis

🛠 Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib / Seaborn
