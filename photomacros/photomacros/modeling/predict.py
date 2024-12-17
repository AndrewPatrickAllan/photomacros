from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()




# imported ourselves --------
import torch
from torch.utils.data import DataLoader
from photomacros.modeling.train import load_data  # Importing own existing load_data function from train.py 
# from torchvision import datasets, transforms
# import config
# from photomacros import dataset
# import random
# -------------------


 

def perform_inference(model_path: Path, 
                      input_path: Path, 
                      predictions_path: Path, 
                      batch_size: int = 32):
    """
    Perform inference on the test dataset using a trained model and save predictions to a CSV file.

    Args:
        model_path (Path): Path to the trained model file (.pkl or .pth).
        input_data_dir (Path): Path to the input data directory (raw or processed images).
        predictions_path (Path): Path to save the predictions CSV file.
        batch_size (int): Batch size for DataLoader.
    """
    

    # Step 1: Load the trained model
    #       (loading a model and loading data are inherently different tasks)
    logger.info(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    logger.success("Model loaded successfully.")

    # Step 2: Load test dataset using the existing load_data function
    logger.info(f"Loading test data from {input_path}...")
    _, _, test_loader = load_data(input_path)  # Get the test DataLoader only
    logger.success("Test data loaded successfully.")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",

    input_path: Path = PROCESSED_DATA_DIR, # added ourselves, copied from train.py 
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")

    perform_inference(model_path,  input_path,  predictions_path, batch_size = 32)
    
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
