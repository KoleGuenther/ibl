import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import learning_curve


def evaluate_model(model, X_test, y_test):
    """
    Update Log:
    10/15/24 Combines all predictions between the confusion matrix, part 1 and part 2, as well as making the model predict. Added Docstrings

    Function: Evaluates the model with a bunch of different stats.

    Parameters:
    - model: The model.
    - X_test: The feature matrix for the test set.
    - y_test: The labels for the test set.

    Returns:
    - dict: A dictionary containing different evaluation scores and the confusion matrix.
    """
    # Predict probabilities
    predict_prob_mlp = model.predict_proba(X_test)[:, 1]

    # Apply fixed threshold to probabilities to get binary predictions
    threshold = 0.45
    predictions = (predict_prob_mlp >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    # Creates a matrix to show the types of guesses, TN, FP, FN, TP.
    # We want as many TN and TP as possible, FP and FN are incorrect guesses.
    # FN is REALLY BAD for a flood prediction, FP is bad but not as bad.
    # FN means that a flooded area was predicted not to be flooded, and it was.
    conf_matrix = confusion_matrix(y_test, predictions)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }, predict_prob_mlp


def output_evaluation_results(evaluation_results):
    """
    Update Log:
    10/15/24 Combined code that showcased Accuracy, Precision, Recall, and F1-Score.

    Function: Show model performance all in one place that is easy to understand.
    Explanation: Accuracy: Pure accuracy (doesn't always show the full picture, this is why other statistics are needed to judge performance)
                 Precision: proportion of correctly predicted positive guesses out of all positive predictions TP/(TP + FP) "Of all the samples that the model predicted as "flooding" (positives), how many were correct?"
                 Recall: proportion of correctly predicted positive guesses out of all instances that are actually positive TP/(TP + FN) "Of all the actual flooding that occurred, how many did the model correctly identify as floods?"
                 F1-Score: combines both precision and recall into a single metric, if there is an imbalanced performance between them, it is more heavily reflected in a F1 score.

    Parameters:
    - evaluation_results: A dictionary containing evaluation metrics.
    """
    print("\nEvaluation Metrics:")
    print(f"{'-'*24}")
    print(f"Accuracy       : {evaluation_results['Accuracy']:.2%}")
    print(f"Precision      : {evaluation_results['Precision']:.2%}")
    print(f"Recall         : {evaluation_results['Recall']:.2%}")
    print(f"F1 Score       : {evaluation_results['F1 Score']:.2%}")
    print(f"{'-'*24}\n")


def output_confusion_matrix(conf_matrix):
    """
    Update Log:
    10/15/24 Created from old code to make a nice looking confusion matrix to showcase more readably.

    Function: Shows True Negatives, False Positives, False Negatives, and True Positives to check on model performance.
    Explanation: This basically shows where the model could do better, many True Negatives and True Positives is GOOD!
                 False Positives and False Negatives indicate that the model was incorrect about non-flooding and flooding respectively.
                 False Negatives is BAD, this means there was flooding and the model incorrectly predicted that there was no flooding.
                 False Positives are just false alarms. Try to keep these low, but some is reasonable for a warning system.

    Returns: A confusion matrix with the amount of tn, fp, fn, and tps.
    """
    tn, fp, fn, tp = conf_matrix.ravel()
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")


# Calculate VIF (variance inflation factor) to detect if a feature is correlated with other features.
# Helps to choose if a feature should be removed from neural network to improve performance.
# Helps to remove bad trends and noise that the model might train after.
def calculate_vif(df, features):
    """
    Update Log:
    10/15/24 Moved from main.py to evaluation.py to simplify. Added Docstrings

    Function: Calculates the Variable Inflation Factor
    Explanation: This essentially shows how much variables relate to each other, high values (above 10) indicates there could be problems.
                 High VIF indicates multicollinearity, meaning that features are highly correlated with each other.
                 This can make the model focus too heavily on a few features and increase the risk of overfitting.

    Returns: A table showing each feature and its corresponding VIF Value
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]

    # Print VIF values without the index
    print("\nVariance Inflation Factor (VIF) Data:")
    print(vif_data.to_string(index=False))

    return vif_data


def plot_roc_curve(y_test, predict_prob_mlp):
    """
    Update Log:
    10/15/24 Moved from main.py to evaluation.py. Added Docstrings

    Function: Plots the ROC Curve.
    Explanation: ROC Curve shows how well the model distinguishes between positive (flood) and negative (no flood) cases.
                 A curve closer to the top-left corner means the model is doing a good job at predicting each.
                 If the curve is close to the diagonal line, it means the model is guessing. (AKA NOT GOOD)

    Parameters:
    - y_test: The true labels of the test set.
    - predict_prob_mlp: Predicted probabilities of the positive class for the test set.

    Returns: A pop-up ROC Curve Graph
    """
    fpr, tpr, _ = metrics.roc_curve(y_test, predict_prob_mlp)
    auc = metrics.roc_auc_score(y_test, predict_prob_mlp)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curve(model, X, y):
    """
    Update Log:
    10/15/24 Moved from main.py to evaluation.py. Added Docstrings

    Function: Plots Learning curve.
    Explanation: Plots the Learning Curve to check on model learning; similar training and validation scores means good learning, while a gap indicates problems.
                 Detects overfitting, if validation scores are low, and train scores are high. Model may be overfitting.

    Parameters:
    - model: The model
    - X: Features
    - y: Labels

    Returns: A pop-up Learning Curve Graph
    """
    train_sizes = np.linspace(0.001, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=5)
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()
