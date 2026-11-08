# Bank Marketing Subscription Prediction Using Machine Learning

## 1. Project Overview

This project applies machine learning classification algorithms to predict whether a customer will subscribe to a bank term deposit based on their demographic information, financial details, and previous marketing campaign interactions.

Five classification algorithms are implemented and compared using multiple evaluation metrics. An interactive Streamlit application is also developed to allow users to select a model, upload test data, and view its prediction performance.

## 2. Dataset Description

**Dataset:** Bank Marketing – Bank Additional Full Dataset

**Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- Total instances: 41,188
- Total original columns: 21 (20 input features and 1 target variable)
- Target variable: `y`
- Target classes: `yes` and `no`

The dataset contains information about bank customers and their interactions with direct marketing campaigns.

The `duration` feature was excluded from model training because it represents the duration of a marketing call and is not available before the call takes place. Including it could introduce data leakage.

Therefore, **19 input features** were used for model development.

## 3. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

A common preprocessing pipeline was used for all models:
- Missing-value imputation.
- Standardization of numerical features.
- One-hot encoding of categorical features.

The dataset was divided into training and testing sets using an 80:20 split with stratified sampling.

## 4. Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8009 | 0.3679 | 0.6466 | 0.4689 | 0.4011 |
| Support Vector Machine | 0.8331 | 0.8007 | 0.3641 | 0.6455 | 0.4656 | 0.3973 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| K-Nearest Neighbor | 0.8972 | 0.7439 | 0.5835 | 0.3050 | 0.4006 | 0.3725 |
| Decision Tree | 0.8468 | 0.6234 | 0.3249 | 0.3341 | 0.3294 | 0.2430 |

### Key Observations

- Logistic Regression achieved the best overall balance, with the highest F1 score, AUC, Recall, and MCC.
- Support Vector Machine performed similarly to Logistic Regression.
- KNN achieved the highest accuracy and precision but had comparatively low recall.
- Naive Bayes showed relatively good recall but lower precision.
- Decision Tree produced the weakest overall performance among the five models.

Logistic Regression was the strongest overall model based on the combined evaluation metrics. Accuracy alone was not considered sufficient because the dataset has an imbalanced target distribution.

## 5. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

**Application features:**
- Select one of the five trained machine learning models.
- Upload a test dataset in CSV format.
- Preview the uploaded dataset.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Visualize the confusion matrix.
- View the classification report.
- Download prediction results.

## 6. Project Structure

```text
Bank_Marketing_ML_Assignment/
│
├── bank_marketing_ml.ipynb
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── bank-additional-full.csv
├── model_comparison_results.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```


## 7. Installation and Execution

### Prerequisites

- Python 3.x
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook

Open `bank_marketing_ml.ipynb` and execute the cells sequentially to perform data preprocessing, model training, evaluation, and model saving.

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

The default test dataset is `test_data.csv`. Alternatively, users can upload their own compatible test dataset.

## 8. Links

- **GitHub Repository:** https://github.com/BITS-MTech-DataScience/Bank-Marketing-ML-Assignment
- **Live Streamlit Application:** [Add your deployed application URL]

## 9. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms for predicting bank term deposit subscriptions.

The experimental results indicate that Logistic Regression provides the best overall balance of classification performance for this dataset. The Streamlit application makes the trained models accessible through an interactive interface for model selection, evaluation, and prediction.
