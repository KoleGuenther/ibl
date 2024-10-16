import joblib
import pandas as pd


def save_model(model, file_path):
    """
    Update Log:
    10/15/24 Added to save a model for safe keeping (backup)

    Saves the trained model to the specified file path.

    Parameters:
    - model: The model.
    - file_path: Path where the model will be saved.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def save_predictions(predictions, test_index, output_file):
    """
    Update Log:
    10/15/24 Moved from prediction.py to util.py

    Store predictions and saves them to a csv file in data.

    Parameters:
    - predictions: Predicted labels for the test set.
    - test_index: The index for the test samples.
    - output_file: The file name where the predictions should be saved.
    """
    predictions_df = pd.DataFrame({'Index': test_index, 'Predicted Label': predictions})
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
