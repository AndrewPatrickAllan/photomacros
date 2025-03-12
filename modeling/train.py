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
sys.path.append(os.path.abspath('/Users/allan/Documents/GitHub/photomacros'))

from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, MEAN, STD, BATCH_SIZE, NUM_EPOCHS, initial_image_size, max_image_size, patience
import torchvision.models as models
# Additional imports for PyTorch and data handling
import torch
from torchvision import datasets, transforms

from photomacros import dataset  # Custom dataset module
import random
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset
from collections import defaultdict
from collections import Counter
# Typer CLI application
app = typer.Typer()

torch.backends.mps.allow_tf32 = True
def get_augmentation_transforms(image_size):
    """
    Define and return data augmentation transformations for training.

    Returns:
        torchvision.transforms.Compose: A sequence of augmentations to apply to training data.
    """
    return transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Less rotation (better for natural images)
    transforms.RandomHorizontalFlip(),  
    transforms.RandomResizedCrop(image_size),  # Less aggressive cropping
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Less extreme changes
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # Less blur effect
    #transforms.RandomErasing(p=0.05, scale=(0.02, 0.08)),  # Less frequent and smaller erasing
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# Define a separate function to replace the lambda
# def process_ten_crop(crops):
#     """Convert each crop to a tensor and normalize it."""
#     return torch.stack([transforms.Normalize(mean=MEAN, std=STD)(transforms.ToTensor()(crop)) for crop in crops])

def get_validation_transforms(image_size):
    """
    Define and return transformations for validation

    Returns:
        torchvision.transforms.Compose: Transformations to apply to validation and test data.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.12)),  # Resize slightly larger to keep aspect ratio
        transforms.CenterCrop(image_size),  # Crop to exact size
        transforms.ToTensor(),              
        transforms.Normalize(mean=MEAN, std=STD)
        ]) 


def split_data(input_data_dir,num_classes, train_ratio=0.7, val_ratio=0.2, test_ratio=0.0):
    """
    Split the dataset into training, validation, and testing sets while preserving class ratios.

    Args:
        input_data_dir (Path): Path to the dataset directory.
        train_ratio (float): Fraction of the dataset for training.
        val_ratio (float): Fraction of the dataset for validation.
        test_ratio (float): Fraction of the dataset for testing.

    Returns:
        tuple: Training, validation, and testing datasets.
    """
    dataset = datasets.ImageFolder(input_data_dir)  # Load dataset without transform
    torch.manual_seed(42)
# Get class names (folder names)
    class_names = dataset.classes  


# Initialize a dictionary to store one image per class
    seen_classes = {}

# Iterate through dataset
    for image_path, label in dataset.samples:
        folder_name = image_path.split("/")[-2]  # Extract folder name (label)
    
    # Only print the first image for each class
        if folder_name not in seen_classes:
            seen_classes[folder_name] = image_path
            print(f"Class: {label} | Image Path: {image_path}")

        # Optional: If you want to display the image (uncomment below line)
        # from PIL import Image
        # img = Image.open(image_path)
        # img.show()  # This will pop up the image

    # Stop after printing one image for each class
        if len(seen_classes) == len(class_names):
            break

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    # Randomly select a subset of classes
    selected_classes = random.sample(class_indices.keys(), num_classes)

    # Filter the indices to only include the selected classes
    filtered_class_indices = {label: indices for label, indices in class_indices.items() if label in selected_classes}
    train_indices, val_indices, test_indices = [], [], []

    # Split each class separately
    for label, indices in filtered_class_indices.items():
        num_samples = len(indices)
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size  # Ensure total matches

        # Shuffle indices before splitting
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(len(indices))]  # Random shuffle

        # Assign indices to each set
        train_indices.extend(indices[:train_size].tolist())
        val_indices.extend(indices[train_size:train_size + val_size].tolist())
        test_indices.extend(indices[train_size + val_size:].tolist())

    return dataset, train_indices, val_indices, test_indices

def update_optimizer_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_data(input_data_dir,num_classes, image_size):
    """
    Load the dataset, apply transformations, and save test data for inference.

    Args:
        input_data_dir (Path): Path to the input dataset directory.

    Returns:
        tuple: DataLoaders for training, validation, and testing datasets.
    """
    dataset, train_indices, val_indices, test_indices = split_data(input_data_dir,num_classes)

    # Apply transformations **after splitting**
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    # """
    # Prints the number of images per class in the original dataset and each split.
    # """
    # # Get all class labels
    # all_labels = [dataset.samples[idx][1] for idx in range(len(dataset))]
    # train_labels = [dataset.samples[idx][1] for idx in train_indices]
    # val_labels = [dataset.samples[idx][1] for idx in val_indices]
    # test_labels = [dataset.samples[idx][1] for idx in test_indices]

    # # Count occurrences of each class
    # original_counts = Counter(all_labels)
    # train_counts = Counter(train_labels)
    # val_counts = Counter(val_labels)
    # test_counts = Counter(test_labels)

    # print("\nClass Distribution:")
    # print(f"{'Class':<10}{'Original':<10}{'Train':<10}{'Val':<10}{'Test':<10}")
    # print("=" * 50)

    # for class_idx in sorted(original_counts.keys()):
    #     print(f"{class_idx:<10}{original_counts[class_idx]:<10}"
    #           f"{train_counts[class_idx]:<10}{val_counts[class_idx]:<10}"
    #           f"{test_counts[class_idx]:<10}")
    # Set transforms **on subsets, not new ImageFolder instances**
    train_dataset.dataset.transform = get_augmentation_transforms(image_size)
    val_dataset.dataset.transform = get_validation_transforms(image_size)
    test_dataset.dataset.transform = get_validation_transforms(image_size)

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


def get_model_architecture(num_classes):
    """
    Define and return the model architecture.

    Args:
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
    #model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)  # Load pretrained model
    #num_features = model.classifier.in_features  # Get the number of input features to the classifier
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze last few layers for fine-tuning
    # for param in list(model.layer4.parameters()) + list(model.fc.parameters()):
    #     param.requires_grad = True

    # # Replace the classifier with a new fully connected layer
    # model.classifier = torch.nn.Sequential(
    #     torch.nn.Linear(num_features, 512),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(512,256),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Dropout(0.5),
    #     torch.nn.Linear(256,num_classes)
    # )

    # return model

    model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)  
    num_features = model.classifier.in_features  # Get the number of input features to the classifier

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    #for param in model.features[-2:].parameters():
        #param.requires_grad = True
    # Replace classifier with a new one
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.5),  # Dropout after first layer

        torch.nn.Linear(512, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(256),
        torch.nn.Dropout(0.5),  # Another Dropout here

        #torch.nn.Linear(256, 128),
        #torch.nn.ReLU(inplace=True),
        #torch.nn.BatchNorm1d(128),
        #torch.nn.Dropout(0.7),  # Additional Dropout layer

        torch.nn.Linear(256, num_classes)  # Output layer
    )


    return model





def evaluate_validation_loss(val_loader, model, criterion):
    """
    Evaluate the model's loss and accuracy on the validation dataset

    Args:
        val_loader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The trained model.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss).

    Returns:
        tuple: (Average validation loss, Top-1 accuracy percentage, Top-5 accuracy percentage)
    """
    device = torch.device("mps")  # Modify to your device if needed (e.g., cuda, cpu)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    num_batches = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
            
            # Forward pass through the model
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate validation loss

            # Compute Top-1 and Top-5 accuracy
            _, pred_top1 = outputs.topk(1, dim=1)  # Get Top-1 prediction
            _, pred_top5 = outputs.topk(5, dim=1)  # Get Top-5 prediction

            # Top-1 accuracy: Check if predicted class matches the true class
            correct_top1 += (pred_top1 == labels.view(-1, 1)).sum().item()

            # Top-5 accuracy: Check if true class is in top 5 predicted classes
            correct_top5 += (pred_top5 == labels.view(-1, 1)).sum().item()

            total += labels.size(0)
            num_batches += 1

    # Average validation loss
    avg_loss = val_loss / num_batches if num_batches > 0 else 0

    # Top-1 and Top-5 accuracy
    top1_acc = (correct_top1 / total) * 100 if total > 0 else 0
    top5_acc = (correct_top5 / total) * 100 if total > 0 else 0

    return avg_loss, top1_acc, top5_acc  # Return all values: loss, top-1, and top-5 accuracies




def train_model(
        # train_loader, val_loader, 
        input_path, 
        initial_image_size, max_image_size, patience):
    """
    Train the model using the training DataLoader. Increase image size if loss stagnates.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        initial_image_size (int): Initial size of images (default is 224).
        max_image_size (int): Maximum image size to increase to (default is 256).
        patience (int): Number of epochs without improvement before increasing image size.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Automatically detect the best device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.success(f"Using device: {device}")
    num_classes = 101#len(train_loader.dataset.dataset.classes)
    image_size=initial_image_size
    train_loader, val_loader, test_loader = load_data(input_path,num_classes,image_size=image_size)

    # Model setup
    
    model = get_model_architecture(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001,betas=(0.9,0.999),weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=7)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epoch_since_last_improvement = 0
    image_size_increased = False  # Prevent further increases
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        avg_val_loss, top1_acc, top5_acc = evaluate_validation_loss(val_loader, model, criterion)
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}, Top-1 Acc = {top1_acc:.2f}, Top-5 Acc = {top5_acc:.2f}")
        #scheduler.step(avg_val_loss)

        # If the validation loss has improved, save the model state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            logger.info("Validation loss improved. Model saved.")
            epoch_since_last_improvement = 0
        else:
            patience_counter += 1
            epoch_since_last_improvement += 1
            logger.info(f"No improvement for {patience_counter} epochs.")


# Increase image size & unfreeze layers only once after epoch 15
        if epoch >= 10 and not image_size_increased and image_size < max_image_size:
            new_image_size = min(image_size + 200, max_image_size)  # Ensure it doesn't exceed max
            logger.info(f"Increasing image size from {image_size} to {new_image_size} and unfreezing last 4 layers.")
            
            image_size = new_image_size
            image_size_increased = True  # Prevent further increases

            # Unfreeze last 4 layers
            for param in model.features[-4:].parameters():
                param.requires_grad = True

            # Update learning rate
            new_lr = 0.0001  
            update_optimizer_lr(optimizer, new_lr)

            # Update dataset transforms with new image size
            train_loader.dataset.transform = get_augmentation_transforms(image_size=image_size)
            val_loader.dataset.transform = get_validation_transforms(image_size=image_size)

    # Load the best model state before returning
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
    trained_model = train_model( input_path, initial_image_size, max_image_size, patience)
    torch.save(trained_model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}.")


if __name__ == "__main__":
    app()