
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Campaign Prediction")
st.write(
    "Compare machine learning models and evaluate their "
    "performance on the uploaded test dataset."
)

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

MODEL_DIR = Path("model")

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

# --------------------------------------------------
# Dataset upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(test_df.head())

    if "y" not in test_df.columns:
        st.error(
            "The uploaded dataset must contain the target "
            "column 'y' to calculate evaluation metrics."
        )
        st.stop()

    # Separate features and target
    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------

    st.header("2. Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose a model",
        list(model_files.keys())
    )

    model_path = MODEL_DIR / model_files[selected_model]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # Validate uploaded features against model features
    required_features = list(model.feature_names_in_)
    missing_features = [
        col for col in required_features
        if col not in X_test.columns
    ]

    if missing_features:
        st.error(
            f"Missing required features: {missing_features}"
        )
        st.stop()

    # Use the exact training feature order
    X_test = X_test[required_features]

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # Obtain scores for the positive class ("yes")
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index("yes")

    if hasattr(classifier, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, positive_index]

    elif hasattr(classifier, "decision_function"):
        decision_scores = model.decision_function(X_test)

        if positive_index == 1:
            y_score = decision_scores
        else:
            y_score = -decision_scores

    else:
        y_score = None

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    st.header("3. Model Evaluation")

    y_true_binary = (y_test == "yes").astype(int)

    if y_score is not None:
        auc = roc_auc_score(y_true_binary, y_score)
    else:
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    col1, col2, col3 = st.columns(3)

    metric_items = list(metrics.items())

    for i, (metric_name, metric_value) in enumerate(metric_items):
        with [col1, col2, col3][i % 3]:
            st.metric(
                label=metric_name,
                value=(
                    f"{metric_value:.4f}"
                    if not np.isnan(metric_value)
                    else "N/A"
                )
            )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["no", "yes"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{selected_model} - Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    st.header("5. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(4))

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    st.header("6. Prediction Summary")

    total = len(y_pred)
    predicted_yes = int(np.sum(y_pred == "yes"))
    predicted_no = int(np.sum(y_pred == "no"))

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", total)
    col2.metric("Predicted Yes", predicted_yes)
    col3.metric("Predicted No", predicted_no)

else:
    st.info(
        "Please upload test_data.csv to begin model evaluation."
    )
