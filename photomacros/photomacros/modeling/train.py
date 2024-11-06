from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE



# imported ourselves --------
import torch
from torchvision import datasets, transforms
# import config
from photomacros import dataset
import random
# -------------------



app = typer.Typer()


# Set random seed for reproducibility
random.seed(46)


# Define on-the-fly augmentations and normalization for training data
def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),           # Rotate images slightly
        transforms.RandomHorizontalFlip(p=0.5),          # Flip images horizontally with a 50% chance
        transforms.RandomResizedCrop(IMAGE_SIZE,  # Crop and resize to standard dimensions
                                     scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3,           # Adjust brightness
                               contrast=0.3,
                               saturation=0.3),
        transforms.ToTensor(),                           # Convert image to tensor
        transforms.Normalize(mean=MEAN, std=STD)  # Normalize
    ])

# Define validation and testing transformations without augmentations
def get_validation_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

# STEP 1 - Splits data into 70% training, 15% validation, 15% testing
def split_data(input_path):
    image_paths = list(Path(input_path).rglob("*.jpg"))  # Get all image paths
    dataset_size = len(image_paths)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)   # 15% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 15% for testing
    
    # Randomly split datasets
    train_dataset, val_dataset, test_dataset = random_split(image_paths, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


# Load datasets and create DataLoaders
def load_data(input_data_dir):
    train_dataset, val_dataset, test_dataset = split_data(input_data_dir)

    # STEP 2 - Create DataLoaders with augmentations
    train_loader = DataLoader(
        datasets.ImageFolder(input_data_dir, transform=get_augmentation_transforms()), 
        batch_size=BATCH_SIZE, 
        sampler=train_dataset,
        shuffle=False  # Avoid shuffling as we're using a sampler
    )

    val_loader = DataLoader(
        datasets.ImageFolder(input_data_dir, transform=get_validation_transforms()), 
        batch_size=BATCH_SIZE, 
        sampler=val_dataset,
        shuffle=False  # Avoid shuffling as we're using a sampler
    )

    test_loader = DataLoader(
        datasets.ImageFolder(input_data_dir, transform=get_validation_transforms()), 
        batch_size=BATCH_SIZE, 
        sampler=test_dataset,
        shuffle=False  # Avoid shuffling as we're using a sampler
    )

    return train_loader, val_loader, test_loader

# Model training loop
def train_model(train_loader):
    model = ...  # Initialize your model here
    optimizer = ...  # Define optimizer (e.g., Adam, SGD)
    criterion = ...  # Define loss function (e.g., CrossEntropyLoss)
    
    model.train()  # Set model to training mode
    for epoch in range(NUM_EPOCHS):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    print("Training complete")



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
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(" Begining training  ")

    logger.info(" beep bop boop ")
    
    logger.info(" we are loading training data ")
    train_loader, val_loader, test_loader = load_data(input_path)

    # logger.info(" we are training the model ")
    # train_model(train_loader)


    
    logger.success(" End training ")
    # -----------------------------------------


if __name__ == "__main__":
    app()
