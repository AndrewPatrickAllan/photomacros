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
from photomacros.modeling.train import load_data, get_model_architecture  # Importing own existing load_data function from train.py 
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
    inference is the act of applying the model (using its learned parameters) to input data to predict outcomes.

    Args:
        model_path (Path): Path to the trained model file (.pkl or .pth).
        input_data_dir (Path): Path to the input data directory (raw or processed images).
        predictions_path (Path): Path to save the predictions CSV file.
        batch_size (int): Batch size for DataLoader.
    """
    
    # # Step 1: Initialize the model architecture
    # model = get_model_architecture(IMAGE_SIZE, num_classes)
    
    # # Step 2: Load the trained weights (state_dict) into the model
    # model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    # model.eval()  # Set the model to evaluation mode

 
    # Step 1: Load num_classes from the saved file
    # with open(MODELS_DIR / "num_classes.txt", "r") as f:
    #     num_classes =  int(f.read().strip())
    num_classes = 101
    
    # Step 2: Initialize the model architecture
    logger.info("Initializing the model architecture...")   
    model = get_model_architecture(IMAGE_SIZE, num_classes)
    
    # Step 3: Load the trained weights (state_dict) into the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    logger.success("Model loaded successfully.")

    # Step 4: Load the saved test dataset
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = torch.load(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.success("Test dataset loaded successfully.")

    # Step 5: Perform inference
    logger.info("Performing inference...")
    predictions = []

    with torch.no_grad():  # Disable gradient computation for inference
        for images, labels in test_loader:  # Load images and labels
            outputs = model(images)
            predicted_classes = outputs.argmax(dim=1)  # For classification tasks
            predictions.extend(predicted_classes.cpu().numpy())

    # Step 6: Save predictions
    logger.info(f"Saving predictions to {predictions_path}...")
    torch.save(predictions, predictions_path)
    logger.success(f"Predictions saved to {predictions_path}.")




def save_test_labels(predictions, test_dataset_path: Path, output_path: Path):
    # Load test data
    test_data = torch.load(test_dataset_path)

    # Assuming test_data contains images and labels (image, label pairs)
    # predictions should match the labels from the test_data
    test_df = pd.DataFrame({
        'image': [item[0] for item in test_data],
        'predicted_label': predictions
    })

    # Save to CSV
    test_df.to_csv(output_path, index=False)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv", # also in train.py 
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.pt",  # we changed from .csv to .pt given the high number of data in food 101. 
    input_path: Path = PROCESSED_DATA_DIR, # added ourselves, copied from train.py, 
    test_labels_path = Path(PROCESSED_DATA_DIR) / "test_labels.pt" # was previously in csv burt was just image matrix and labels (index) so not human readbale anyway. 
):

    logger.info("Performing inference for model...")
    perform_inference(model_path,  input_path,  predictions_path, batch_size = BATCH_SIZE)
    logger.success("Inference complete.")


    # Load predictions
    predictions = torch.load(predictions_path)

    logger.info(f"Save predictions with corresponding labels to {test_labels_path}...") 
    # Save predictions with corresponding labels
    save_test_labels(predictions, test_data_path, test_labels_path)
    logger.success(f"Test labels saved to {test_labels_path}")



if __name__ == "__main__":
    app()