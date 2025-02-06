import torch
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from photomacros.config import PROCESSED_DATA_DIR, REPORTS_DIR


# we have created evaluate.py to evaluate the model predictions generated in predict.py 

def load_predictions(predictions_path):
    """
    Load predictions saved as a PyTorch tensor.
    """
    return torch.load(predictions_path)

def load_ground_truth(test_labels_path):
    """
    Load ground truth labels for the test dataset.
    """
    return pd.read_csv(test_labels_path)  # Assuming labels are in a CSV file

def evaluate_predictions(predictions, ground_truth_labels):
    """
    Evaluate predictions against ground truth labels.
    """
    # Extract predicted and actual labels
    y_pred = predictions  # Assuming predictions are saved as class indices
    y_true = ground_truth_labels["label"].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return accuracy, report

def main(
    
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.pt",  # from predict.py
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.pt",  # from predict.py
    metrics_output_path: Path = REPORTS_DIR / "metrics.json"
):
    """
    Main function to evaluate predictions and save metrics.
    """
    print("Loading predictions...")
    predictions = load_predictions(predictions_path)

    print("Loading ground truth labels...")
    ground_truth_labels = load_ground_truth(test_labels_path)

    print("Evaluating predictions...")
    accuracy, report = evaluate_predictions(predictions, ground_truth_labels)

    print(f"Accuracy: {accuracy:.4f}")
    
    # Save the evaluation results
    metrics = {"accuracy": accuracy, "classification_report": report}
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure reports dir exists

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    main()