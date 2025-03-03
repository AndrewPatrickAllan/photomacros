"""
Script for training a machine learning model using PyTorch.

This script includes functionality to:
- Define and apply data augmentation transformations.
- Split datasets into training, validation, and testing sets.
- Define the model architecture.
- Train the model using a specified number of epochs.
- Save the trained model for later inference or evaluation.

Modules:
    - Path: Provides easy manipulation of filesystem paths.
    - typer: Facilitates the creation of CLI commands.
    - logger (loguru): Adds advanced logging capabilities.
    - tqdm: Displays progress bars for loops.
    - torch and torchvision: PyTorch libraries for deep learning.
    - photomacros: Custom dataset utilities.
"""
import sys
import os
sys.path.append(os.path.abspath('~/Documents/GitHub/photomacros'))

from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_EPOCHS
import torchvision.models as models
# Additional imports for PyTorch and data handling
import torch
from torchvision import datasets, transforms

from photomacros import dataset  # Custom dataset module
import random
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset

# Typer CLI application
app = typer.Typer()


def get_augmentation_transforms():
    """
    Define and return data augmentation transformations for training.

    Returns:
        torchvision.transforms.Compose: A sequence of augmentations to apply to training data.
    """
    return transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Less rotation (better for natural images)
    transforms.RandomHorizontalFlip(),  
    transforms.RandomResizedCrop(IMAGE_SIZE),  # Less aggressive cropping
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Less extreme changes
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 0.7)),  # Less blur effect
    transforms.RandomErasing(p=0.05, scale=(0.01, 0.03)),  # Less frequent and smaller erasing
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


def get_validation_transforms():
    """
    Define and return transformations for validation and testing.

    Returns:
        torchvision.transforms.Compose: Transformations to apply to validation and test data.
    """
    return transforms.Compose([
        transforms.Resize(255),                  # Resize image to the specified size
        transforms.CenterCrop(224), 
        transforms.ToTensor(),                          # Convert image to tensor
        transforms.Normalize(mean=MEAN, std=STD)        # Normalize image tensor
    ])


def split_data(input_data_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    """
    Split the dataset into training, validation, and testing sets.

    Args:
        input_data_dir (Path): Path to the dataset directory.
        train_ratio (float): Fraction of the dataset for training.
        val_ratio (float): Fraction of the dataset for validation.
        test_ratio (float): Fraction of the dataset for testing.

    Returns:
        tuple: Training, validation, and testing datasets.
    """
    dataset = datasets.ImageFolder(input_data_dir)  # No transform here
    torch.manual_seed(42)

    # Compute sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure sum matches

    # Split the dataset
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        list(range(dataset_size)), [train_size, val_size, test_size]
    )

    return dataset, train_indices, val_indices, test_indices

def load_data(input_data_dir):
    """
    Load the dataset, apply transformations, and save test data for inference.

    Args:
        input_data_dir (Path): Path to the input dataset directory.

    Returns:
        tuple: DataLoaders for training, validation, and testing datasets.
    """
    dataset, train_indices, val_indices, test_indices = split_data(input_data_dir)

    # Apply transformations **after splitting**
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Set transforms **on subsets, not new ImageFolder instances**
    train_dataset.dataset.transform = get_augmentation_transforms()
    val_dataset.dataset.transform = get_validation_transforms()
    test_dataset.dataset.transform = get_validation_transforms()

    # Save datasets
    torch.save(test_dataset, MODELS_DIR / "test_data.pt")
    torch.save(val_dataset, MODELS_DIR / "val_data.pt")
    torch.save(train_dataset, MODELS_DIR / "train_data.pt")
    logger.success(f"Datasets saved to {MODELS_DIR}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=False)
    return train_loader, val_loader, test_loader


def get_model_architecture(image_size, num_classes):
    """
    Define and return the model architecture.

    Args:
        image_size (int): Input image size (assumes square images).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Sequential: Model architecture.
    """
    # model = torch.nn.Sequential(
    #     torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
    #     torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(64 * (image_size // 4) * (image_size // 4), 128),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(128, num_classes)
    # )
    # return model
    # model = torch.nn.Sequential(
    #     # First Convolutional Block
    #     torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(64),  # Batch normalization
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Second Convolutional Block
    #     torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(128),  # Batch normalization
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Third Convolutional Block
    #     torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(256),  # Batch normalization
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Fourth Convolutional Block (Increased complexity)
    #     torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(512),  # Batch normalization
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Fifth Convolutional Block (Increased complexity)
    #     torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(1024),  # Batch normalization
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Global Average Pooling (to reduce to 1x1 feature map)
    #     torch.nn.AdaptiveAvgPool2d((1, 1)),  # Reduces to 1x1 feature map for each channel

    #     # Flattening
    #     torch.nn.Flatten(),

    #     # Fully Connected Layers (Increased depth)
    #     torch.nn.Linear(1024, 512),
    #     torch.nn.BatchNorm1d(512),  
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.1),

    #     torch.nn.Linear(512, 256),
    #     torch.nn.BatchNorm1d(256),  
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.1),

    #     torch.nn.Linear(256, num_classes)  # Output layer
    # )
    # model = torch.nn.Sequential(
    #     # First Convolutional Block
    #     torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(64),
    #     torch.nn.SiLU(),
    #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Extra conv layer
    #     torch.nn.BatchNorm2d(64),
    #     torch.nn.SiLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),  

    #     # Second Convolutional Block
    #     torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(128),
    #     torch.nn.SiLU(),
    #     torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Extra conv layer
    #     torch.nn.BatchNorm2d(128),
    #     torch.nn.SiLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Third Convolutional Block (Smaller filter size)
    #     torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(256),
    #     torch.nn.SiLU(),
    #     torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(256),
    #     torch.nn.SiLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Fourth Convolutional Block (Reduce number of filters)
    #     torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(384),
    #     torch.nn.SiLU(),
    #     torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
    #     torch.nn.BatchNorm2d(384),
    #     torch.nn.SiLU(),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2),

    #     # Global Average Pooling instead of fully connected layers
    #     torch.nn.AdaptiveAvgPool2d((1, 1)),  
    #     torch.nn.Flatten(),

    #     # Fully Connected Layers
    #     torch.nn.Linear(384, 256),
    #     torch.nn.BatchNorm1d(256),  
    #     torch.nn.SiLU(),
    #     torch.nn.Dropout(0.3),  # Only one dropout here

    #     torch.nn.Linear(256, 128),
    #     torch.nn.BatchNorm1d(128),  
    #     torch.nn.SiLU(),

    #     torch.nn.Linear(128, num_classes)
    # )
    #pretained is better? has many more layers, we will see
    model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)  # Load pretrained model
    num_features = model.classifier.in_features  # Get the number of input features to the classifier
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier with a new fully connected layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512,256),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256,num_classes)
    )

    return model


def evaluate_validation_loss(val_loader, model, criterion):
    """
    Evaluate the model's loss and accuracy on the validation dataset.

    Args:
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The trained model.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss).

    Returns:
        tuple: (Average validation loss, Accuracy percentage)
    """
    device = torch.device("mps")
    model.to(device)
    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():  
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to MPS
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()  
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            num_batches += 1

    avg_loss = val_loss / num_batches if num_batches > 0 else 0  
    accuracy = (correct / total) * 100 if total > 0 else 0  # Accuracy in percentage

    return avg_loss, accuracy
def train_model(train_loader,val_loader):
    """
    Train the model using the training DataLoader.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        torch.nn.Sequential: Trained model.
    """

# Automatically detect the best device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.success(f"Using device: {device}")

    num_classes = len(train_loader.dataset.dataset.classes)
    model = get_model_architecture(IMAGE_SIZE, num_classes).to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()
    # Cosine Annealing LR Scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='max',   # Reduce LR when validation loss decreases
    factor=1/3,   # Reduce LR by a factor of 0.5 (adjust as needed)
    patience=2,   # Wait for 3 epochs without improvement before reducing
    )
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = 0.0

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        val_loss,accuracy = evaluate_validation_loss(val_loader, model, criterion)
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Validation Loss = {val_loss:.4f}, Accuracy = {accuracy}")
        scheduler.step(val_loss)  # Update LR according to Cosine Annealing
        logger.info(f"Updated Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            logger.info("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break  

    model.load_state_dict(best_model_state)
    return model

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR / f"model_{NUM_EPOCHS}epochs_BetterModel_LR_Earlystop_pretrainedDenseNet.pkl"
):
    """
    Main function to train the model and save the trained model.

    Args:
        input_path (Path): Path to the input dataset directory.
        model_path (Path): Path to save the trained model.
    """
    logger.info("Starting training process...")
    train_loader, val_loader, test_loader = load_data(input_path)
    trained_model = train_model(train_loader,val_loader)
    torch.save(trained_model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}.")


if __name__ == "__main__":
    app()