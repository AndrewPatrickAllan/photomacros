from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from torchvision import models 
import torch.nn as nn
import torch.optim as optim
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

    # Randomly split datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset
  
  


# Load datasets and create DataLoaders
def load_data(input_data_dir):
    train_dataset, val_dataset, test_dataset = split_data(input_data_dir)

    # STEP 3 - Create DataLoaders with augmentations
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
    
    logger.success("Train, validation, and test datasets generation complete with labels.")

    return train_loader, val_loader, test_loader
  
# Model training loop
def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        validate_model(val_loader, model, criterion)

    logger.success("Training complete")

def validate_model(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

@app.command()
def main(features_path: Path = PROCESSED_DATA_DIR / "features.csv",
         label_path: Path = PROCESSED_DATA_DIR / "features.csv",
         model_path: Path = PROCESSED_DATA_DIR / "model.pkl",
         input_path: Path = PROCESSED_DATA_DIR):
    
    logger.info("Beginning training...")
    logger.info("Loading training data...")
    train_loader, val_loader, test_loader = load_data(input_path)

    # Define model, optimizer, loss function
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(set([label for _, label in train_loader.dataset])))
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs=10)

    logger.success("Training complete")
    
    logger.success(" End training ")
    # -----------------------------------------



if __name__ == "__main__":
    app()
