import joblib
import os
import numpy as np

from data_preprocessing import load_and_prepare_data, add_elevation_data, handle_missing_values, scale_features
from sklearn.neural_network import MLPClassifier
from util import save_predictions, save_model
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from evaluation import evaluate_model, output_confusion_matrix, calculate_vif, output_evaluation_results, plot_roc_curve, plot_learning_curve


# Data File - Dataset to be trained on
data_file = 'data/finaldata_impervious_elevation.csv'
# Output File - Outputting the results after the model predicts
output_file = 'data/test1.csv'

# Model File - The Saved Model that will try to be loaded if the flag below is set to True
model_file = 'models/101524Model'
# Set flag to decide if a saved model should be used
use_saved_model = False  # True to load the saved model, or False to train a new one


# Preprocessing Data
dataset = load_and_prepare_data(data_file)
dataset = add_elevation_data(dataset)
dataset = handle_missing_values(dataset, -99999)

# Relevant features that will be used
feature_vars = [
    '%_pop_in_very_high_flood_hazard_area',
    '%_wetlands',
    'elevation',
    '%_imperviousarea_15mbuffer'
]
# Features Used By Team:
# feature_vars = ['elevation', '%_residential_population_within300m_busyroadway', '%_busy_roadway_bordered_<25_percent_treebuffer', 'residential_population_within_300m_of_busy_road', 'residential_population_within_300m_of_busy_road_with_less_than_ 25percent_tree_buffer', 'total_pop_under_age_1']


# Scales Features
X = scale_features(dataset, feature_vars)
y = dataset['label']

# Splits data into two groups (As of writing this, its 70% training, 30% testing)
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X, y, dataset.index, test_size=0.30, random_state=42
)


def train_model(data, labels):
    """
    Update Log:
    10/15/24 Moved from model_training to here to simplify files. Converted Comments into Docstrings

    Solver: lbfgs, adam, sgd
    Hidden_Layer_Sizes: Think of this as neurons connecting, and each comma makes a new layer
    Max Iterations: The amount of times the model will try learning (typically higher amount means a better model performance, but takes longer to run).
    Alpha: Regularization
    Tolerance: The model will stop when it is not improving by this amount per iteration.
    Random State: Reproducible but still random
    Learning Rate: How fast it learns

    Function: Trains an MLPClassifier with predefined parameters on the given data.

    Parameters:
    - X: Data for training.
    - y: Labels for training.

    Returns:
    - mlp: Trained model.
    """
    mlp = MLPClassifier(
        solver='lbfgs',
        hidden_layer_sizes=(8, 4,),
        max_iter=2000,
        alpha=0.1,
        activation='relu',
        tol=1e-5,
        random_state=4,
        learning_rate_init=0.001
    )
    mlp.fit(data, labels)
    return mlp


# Model Loading | Model Training
if use_saved_model and os.path.exists(model_file):
    # Load saved model if flag is True and the model file exists
    model = joblib.load(model_file)
    print(f"Model loaded from {model_file}")
else:
    # Checks if the model file already exists (so you aren't stupid and delete the model you want)
    if os.path.exists(model_file):
        user_response = input(f"This file '{model_file}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
        if user_response == 'no':
            print("Model training aborted to avoid overwriting the existing model.")
            exit()
        elif user_response != 'yes':
            print("Invalid input. Model training aborted.")
            exit()

    # Trains a new model
    model = train_model(X_train, y_train)
    # Saves the new model
    save_model(model, model_file)
    print(f"New model trained and saved to {model_file}")

# Evaluates the model on the test data
evaluation_results, predict_prob_mlp = evaluate_model(model, X_test, y_test)

# Predicts probabilities for outcomes and saves predictions to output file
predictions = model.predict(X_test)
save_predictions(predictions, test_index=X_test.index, output_file=output_file)

# Shows Model Performance
output_evaluation_results(evaluation_results)
# Shows Confusion Matrix
confusion_matrix = evaluation_results["Confusion Matrix"]
output_confusion_matrix(confusion_matrix)

# VIF Results
vif_data = calculate_vif(dataset, feature_vars)

# Correlation analysis
# Checks how much a variable has a correlation with Label
correlation_matrix = dataset[feature_vars + ['label']].corr()
correlation_with_label = correlation_matrix['label'].drop('label')
print("\nCorrelation of Features with the Label:")
print(correlation_with_label)

# Variation of Cross-Validation that preserves class distribution, in case a dataset has more "no flood" cases than "flood" cases...
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

# Cross-Validation Scores
# If mean score is similar to accuracy and std dev is low, then the model's performance is consistent and gives a reliable estimate that the model will generalize well to new data.
print(f"\nStratifiedKFold:")
print(f"Cross-Validation Accuracy Scores (for each fold): {cross_val_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cross_val_scores):.4f}")
print(f"Standard Deviation of Cross-Validation Accuracy: {np.std(cross_val_scores):.4f}")

# Plot ROC Curve
plot_roc_curve(y_test, predict_prob_mlp)

# Plot Learning Curve
plot_learning_curve(model, X, y)
