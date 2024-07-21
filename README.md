# Telecom Churn Prediction

## Introduction
Customer churn is a critical issue in the telecom industry, where companies strive to retain customers and reduce turnover. This project aims to predict customer churn using various machine learning techniques. By identifying factors contributing to churn, we can help telecom companies devise strategies to retain their customers.

## Dataset
The dataset used for this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/datazng/telecom-company-churn-rate-call-center-data). It contains 7043 customer records with 21 features.

**Features:**
- `customerID`: Unique customer ID
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1) or not (0)
- `Partner`: Indicates if the customer has a partner (Yes/No)
- `Dependents`: Indicates if the customer has dependents (Yes/No)
- `tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Indicates if the customer has phone service (Yes/No)
- `MultipleLines`: Indicates if the customer has multiple lines (Yes/No/No phone service)
- `InternetService`: Customer’s internet service provider (DSL/Fiber optic/No)
- `OnlineSecurity`: Indicates if the customer has online security (Yes/No/No internet service)
- `OnlineBackup`: Indicates if the customer has online backup (Yes/No/No internet service)
- `DeviceProtection`: Indicates if the customer has device protection (Yes/No/No internet service)
- `TechSupport`: Indicates if the customer has tech support (Yes/No/No internet service)
- `StreamingTV`: Indicates if the customer has streaming TV (Yes/No/No internet service)
- `StreamingMovies`: Indicates if the customer has streaming movies (Yes/No/No internet service)
- `Contract`: Type of customer contract (Month-to-month/One year/Two year)
- `PaperlessBilling`: Indicates if the customer uses paperless billing (Yes/No)
- `PaymentMethod`: Customer’s payment method (Mailed check/Electronic check/Credit card/Bank transfer)
- `MonthlyCharges`: Monthly charges of the customer
- `TotalCharges`: Total charges to the customer
- `Churn`: Target variable indicating if the customer churned (Yes/No)

## Problem Statement
The goal of this project is to predict customer churn based on various attributes of the customer. By understanding the factors leading to churn, telecom companies can take proactive measures to improve customer retention.

## Exploratory Data Analysis (EDA)
EDA involves understanding the dataset, identifying patterns, and visualizing data to gain insights. Key steps include:
- Data cleaning: Handling missing and duplicate values
- Descriptive statistics: Summarizing the data
- Correlation analysis: Identifying relationships between features
- Visualization: Using plots to understand the distribution and relationships of features

## Feature Engineering
Feature engineering transforms raw data into meaningful features for model building:
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Handling imbalanced data using techniques like SMOTE (Synthetic Minority Over-sampling Technique)
- Dimensionality reduction using PCA (Principal Component Analysis)

## Modeling
We employ various machine learning algorithms to build predictive models:
1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **XGBoost**
4. **Multi-layered Perceptron (MLP)**

## Model Evaluation
Models are evaluated using metrics such as accuracy, precision, recall, and F1-score. The focus is on F1-score due to the imbalanced nature of the dataset. Cross-validation and hyperparameter tuning are also performed to optimize model performance.

## Conclusion
The project identifies key factors influencing customer churn and provides actionable insights for telecom companies. By leveraging predictive models, companies can target at-risk customers and implement retention strategies effectively.

## Usage
1. Ensure the dataset is placed in the `Input` folder.
2. Run the Jupyter notebook or Python script to perform EDA, feature engineering, and modeling.
3. Review the results and model outputs to understand the churn patterns.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository, create a branch, and submit a pull request.

Feel free to customize this text according to your project's specifics and requirements.
