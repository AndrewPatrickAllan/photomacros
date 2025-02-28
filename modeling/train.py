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

from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_EPOCHS

# Additional imports for PyTorch and data handling
import torch
from torchvision import datasets, transforms
from photomacros import dataset  # Custom dataset module
import random
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast

# Typer CLI application
app = typer.Typer()

# Set random seed for reproducibility
random.seed(46)


def get_augmentation_transforms():
    """
    Define and return data augmentation transformations for training.

    Returns:
        torchvision.transforms.Compose: A sequence of augmentations to apply to training data.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),           # Rotate images by up to 15 degrees
        transforms.RandomHorizontalFlip(p=0.5),          # Flip images horizontally with a 50% chance
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),  # Randomly crop and resize
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Adjust image color properties
        transforms.ToTensor(),                           # Convert image to tensor
        transforms.Normalize(mean=MEAN, std=STD)         # Normalize image tensor
    ])


def get_validation_transforms():
    """
    Define and return transformations for validation and testing.

    Returns:
        torchvision.transforms.Compose: Transformations to apply to validation and test data.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),                  # Resize image to the specified size
        transforms.ToTensor(),                          # Convert image to tensor
        transforms.Normalize(mean=MEAN, std=STD)        # Normalize image tensor
    ])


def split_data(input_data_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
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
    dataset = datasets.ImageFolder(input_data_dir, transform=None)
    torch.manual_seed(42) 
    # Compute sizes for splits
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def evaluate_validation_loss(val_loader, model, criterion):
    """
    Evaluate the model's loss on the validation dataset.

    Args:
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The trained model.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss).

    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # No need to compute gradients
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate loss per batch
            num_batches += 1

    return val_loss / num_batches if num_batches > 0 else 0  # Return average loss per batch
def load_data(input_data_dir):
    """
    Load the dataset, apply transformations, and save test data for inference.

    Args:
        input_data_dir (Path): Path to the input dataset directory.

    Returns:
        tuple: DataLoaders for training, validation, and testing datasets.
    """
    train_dataset, val_dataset, test_dataset = split_data(input_data_dir)

    # Apply transformations
    train_dataset.dataset.transform = get_augmentation_transforms()
    val_dataset.dataset.transform = get_validation_transforms()
    test_dataset.dataset.transform = get_validation_transforms()

    # Save datasets for future use
    torch.save(test_dataset, MODELS_DIR / "test_data.pt")
    torch.save(val_dataset, MODELS_DIR / "val_data.pt")
    torch.save(train_dataset, MODELS_DIR / "train_data.pt")
    logger.success(f"Datasets saved to {MODELS_DIR}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    model = torch.nn.Sequential(
    # First Convolutional Block
    torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.BatchNorm2d(32),  # Batch normalization
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    # Second Convolutional Block
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.BatchNorm2d(64),  # Batch normalization
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    # Third Convolutional Block
    torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    torch.nn.BatchNorm2d(128),  # Batch normalization
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    # Fourth Convolutional Block
    torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    torch.nn.BatchNorm2d(256),  # Batch normalization
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),

    # Global Average Pooling
    torch.nn.AdaptiveAvgPool2d((1, 1)),  # Reduces to 1x1 feature map for each channel

    # Flattening
    torch.nn.Flatten(),

    # Fully Connected Layer
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),  # Dropout to prevent overfitting

    # Output Layer
    torch.nn.Linear(128, num_classes)
)

    return model



def train_model(train_loader,val_loader):
    """
    Train the model using the training DataLoader.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        torch.nn.Sequential: Trained model.
    """
    num_classes = len(train_loader.dataset.dataset.classes)
    model = get_model_architecture(IMAGE_SIZE, num_classes)
    class_names = train_loader.dataset.dataset.classes  # Extract class names
    with open(MODELS_DIR / "class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    #print("Class names saved:", class_names)

    # Save number of classes for later use
    with open(MODELS_DIR / "num_classes.txt", "w") as f:
        f.write(str(num_classes))

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) #better optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # Learning rate scheduler: Reduce LR if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    patience = 5  # Stop training if no improvement after 'patience' epochs
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = 0.0

        for batch_idx, (images, labels) in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        # Compute validation loss at the end of each epoch
        val_loss = evaluate_validation_loss(val_loader, model, criterion)
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Validation Loss = {val_loss:.4f}")

        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
            best_model_state = model.state_dict()  # Save best model
            logger.info("Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break  # Stop training if no improvement for 'patience' epochs

    # Load best model state before returning
    model.load_state_dict(best_model_state)
    return model
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR / f"model_{NUM_EPOCHS}epochs_BetterModel_LR_Earlystop.pkl"
):
    """
    Main function to train the model and save the trained model.

    Args:
        input_path (Path): Path to the input dataset directory.
        model_path (Path): Path to save the trained model.
    """
    logger.info("Starting training process...")
    print (f"model_{NUM_EPOCHS}epochs_BetterModel_LRScheduler_Earlystop.pkl")
    train_loader, val_loader, test_loader = load_data(input_path)
    trained_model = train_model(train_loader,val_loader)
    torch.save(trained_model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}.")


if __name__ == "__main__":
    app()