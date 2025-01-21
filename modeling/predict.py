from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, test_data_path, BATCH_SIZE


app = typer.Typer()




# imported ourselves --------
import torch
from torch.utils.data import DataLoader
from train import load_data, get_model_architecture  # Importing own existing load_data function from train.py
# from torchvision import datasets, transforms
# import config
# from photomacros import dataset
# import random
# -------------------

def perform_inference(
    model_path: Path,
    test_data_path: Path,
    predictions_path: Path,
    batch_size: int = 32
):
    """
    Perform inference on the test dataset using a trained model and save predictions to a file.

    Args:
        model_path (Path): Path to the trained model file (.pkl or .pth).
        test_data_path (Path): Path to the saved test dataset.
        predictions_path (Path): Path to save the predictions.
        batch_size (int): Batch size for DataLoader.
    """
    # Step 1: Determine the number of classes
    with open(MODELS_DIR / "num_classes.txt", "r") as f:
        num_classes = int(f.read().strip())

    # Step 2: Initialize the model architecture
    logger.info("Initializing the model architecture...")
    model = get_model_architecture(IMAGE_SIZE, num_classes)

    # Step 3: Load the trained model
    logger.info(f"Loading trained model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    logger.success("Model loaded successfully.")

    # Step 4: Load the test dataset
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = torch.load(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.success("Test dataset loaded successfully.")

    # Step 5: Perform inference
    logger.info("Performing inference on the test dataset...")
    predictions = []

    with torch.no_grad():  # Disable gradient computation for inference
        for images, labels in test_loader:  # Only images are needed during inference
            outputs = model(images)
            predicted_classes = outputs.argmax(dim=1)  # Get class predictions
            predictions.extend(predicted_classes.cpu().numpy())

    # Step 6: Save predictions
    logger.info(f"Saving predictions to {predictions_path}...")
    torch.save(predictions, predictions_path)
    logger.success(f"Predictions saved to {predictions_path}.")


def save_test_labels(
    predictions,
    test_data_path: Path,
    output_path: Path
):
    """
    Save predictions and corresponding labels to a file.

    Args:
        predictions (list): List of predicted labels.
        test_data_path (Path): Path to the test dataset file.
        output_path (Path): Path to save the labeled predictions.
    """
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = torch.load(test_data_path)

    logger.info("Creating DataFrame with predictions and ground truth labels...")
    test_labels = [label for _, label in test_dataset]
    test_images = [f"Image_{i}" for i in range(len(test_labels))]  # Placeholder image IDs

    test_df = pd.DataFrame({
        "image": test_images,
        "ground_truth_label": test_labels,
        "predicted_label": predictions
    })

    logger.info(f"Saving labeled predictions to {output_path}...")
    test_df.to_csv(output_path, index=False)
    logger.success(f"Labeled predictions saved to {output_path}.")


@app.command()
def main(
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.pt",
    test_data_path: Path = test_data_path,
    test_labels_output_path: Path = PROCESSED_DATA_DIR / "test_labels.csv"
):
    """
    Main function to perform inference and save predictions with labels.
    """
    logger.info("Starting inference process...")
    perform_inference(
        model_path=model_path,
        test_data_path=test_data_path,
        predictions_path=predictions_path,
        batch_size=BATCH_SIZE
    )

    # Load predictions
    predictions = torch.load(predictions_path)

    # Save predictions with corresponding labels
    logger.info(f"Saving predictions with corresponding labels to {test_labels_output_path}...")
    save_test_labels(predictions, test_data_path, test_labels_output_path)
    logger.success("Inference process completed.")


if __name__ == "__main__":
    app()
