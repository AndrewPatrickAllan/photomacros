from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR
import torch
app = typer.Typer()

def load_data_labels(input_path):
    image_paths = []
    labels = []

    # Traverse the directory
    for label_dir in input_path.iterdir():
        if label_dir.is_dir():  # Check if it's a directory (class label)
            for img_file in label_dir.glob("*.jpg"):  # Load all jpg images
                image_paths.append(img_file)
                labels.append(label_dir.name)  # Use the directory name as the label
               

    logger.info(f"Loaded {len(image_paths)} images with corresponding labels.")
    logger.info("Generating train, val, and test datasets...")

    dataset_size = len(image_paths)

    # Combine images and labels for easy splitting
    dataset = list(zip(image_paths, labels))
    

    # Split the dataset into train, validation, and test sizes
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)  # 15% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 15% for testing

    # Randomly split the dataset
    torch.manual_seed(42)  # Set a random seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.success("Train, validation, and test datasets generation complete with labels.")

    return train_loader, val_loader, test_loader
   

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    label_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = PROCESSED_DATA_DIR / "model.pkl",
    input_path: Path = PROCESSED_DATA_DIR,
    #output_path: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):


    # Create DataLoaders for each dataset
    train_loader,val_loader, test_loader= load_data_labels(input_path)


    logger.success("Train, validation, and test datasets generation complete with labels.")




if __name__ == "__main__":
    app()
