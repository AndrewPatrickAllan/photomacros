import torch
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from photomacros.config import PROCESSED_DATA_DIR, REPORTS_DIR,MODELS_DIR

"""
This script evaluates the model predictions (from predict.py) against ground truth labels
and generates performance metrics like accuracy and classification reports.

The evaluation results are saved as a JSON file for later analysis.
"""

def load_predictions(predictions_path):
    """
    Load predictions saved as a PyTorch tensor file.
    
    Args:
        predictions_path (Path): Path to the saved predictions file (.pt).
    
    Returns:
        torch.Tensor: Predictions as a PyTorch tensor.
    """
    return torch.load(predictions_path)  # Load the .pt file using PyTorch


def load_ground_truth(test_labels_path):
    """
    Load ground truth labels for the test dataset.
    
    Args:
        test_labels_path (Path): Path to the CSV file containing the ground truth labels.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the test labels.
                      Assumes the CSV contains a column "ground_truth_label".
    """
    return pd.read_csv(test_labels_path)  # Load the labels as a DataFrame from the CSV file


def evaluate_predictions(predictions, ground_truth_labels):
    """
    Compare the model's predictions with ground truth labels and calculate evaluation metrics.
    
    Args:
        predictions (torch.Tensor or list): Predicted class indices (output from the model).
        ground_truth_labels (pd.DataFrame): DataFrame containing the ground truth labels. Expects a column "ground_truth_label".
    
    Returns:
        tuple: (accuracy, report)
               - accuracy (float): Overall accuracy score as a decimal.
               - report (dict): Classification report as a dictionary with metrics like precision, recall, f1-score, etc.
    """
    # Extract the true labels from the DataFrame
    # y_true = ground_truth_labels["ground_truth_label"].values  # Ground truth labels as a numpy array
    # print('y_true_new:',y_true)
    y_true = ground_truth_labels["predicted_label"].values  # Ground truth labels as a numpy array
    # print('y_true_old:',y_true)

    y_pred = predictions  # Predicted labels (assumed to be class indices)

    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Generate a detailed classification report (precision, recall, F1-score per class)
    report = classification_report(y_true, y_pred, output_dict=True)

    return accuracy, report  # Return the accuracy and classification metrics


def main(
    predictions_path: Path = MODELS_DIR/ "test_predictions.pt",  # Default path for predictions file
    test_labels_path: Path = MODELS_DIR / "test_labels.csv",      # Default path for test labels CSV
    metrics_output_path: Path = REPORTS_DIR / "metrics.json"             # Default path to save the metrics JSON
):
    """
    Main function to evaluate predictions and save the metrics as a JSON file.
    
    Args:
        predictions_path (Path): path to the .pt file containing the model's predictions.
        test_labels_path (Path): path to the CSV file containing ground truth labels.
        metrics_output_path (Path): path to save the evaluation metrics as a JSON file.
    
    Outputs:
        Saves the evaluation metrics (accuracy and classification report) to the specified JSON file.
    """
    # Step 1: Load predictions
    print("Loading predictions...")
    predictions = load_predictions(predictions_path)  # Load predictions from the .pt file

    # Step 2: Load ground truth labels
    print("Loading ground truth labels...")
    ground_truth_labels = load_ground_truth(test_labels_path)  # Load labels from the CSV file

    # Step 3: Evaluate predictions
    print("Evaluating predictions...")
    accuracy, report = evaluate_predictions(predictions, ground_truth_labels)  # Evaluate metrics

    # Step 4: Print accuracy to the console for quick feedback
    print(f"Accuracy: {accuracy:.4f}")  # Print accuracy to 4 decimal places

    # Step 5: Prepare the metrics as a dictionary
    metrics = {
        "accuracy": accuracy,             # Save the accuracy score
        "classification_report": report  # Save the classification report as a dictionary
    }

    # Step 6: Ensure the output directory exists
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist

    # Step 7: Save metrics to the specified JSON file
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)  # Save the metrics in a pretty JSON format

    # Notify the user where the metrics have been saved
    print(f"Metrics saved to {metrics_output_path}")


# Entry point: Run the script when executed directly
if __name__ == "__main__":
    main()  # Execute the main function with default arguments