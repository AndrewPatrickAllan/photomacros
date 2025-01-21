from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_EPOCHS, test_data_path


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



def get_augmentation_transforms():

    """
    Define on-the-fly augmentations and normalization for training data
    """


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


# Split data into training, validation, and testing sets
from torch.utils.data import random_split
def split_data(input_data_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # Load the dataset
    dataset = datasets.ImageFolder(input_data_dir, transform=None)

    # Compute sizes for splits
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

  
def load_data(input_data_dir):
    # Split the dataset
    train_dataset, val_dataset, test_dataset = split_data(input_data_dir)

    # Apply transforms to each split
    train_dataset.dataset.transform = get_augmentation_transforms()
    val_dataset.dataset.transform = get_validation_transforms()
    test_dataset.dataset.transform = get_validation_transforms()

    # Save the test dataset to a file for inference
    #torch.save(test_dataset, test_data_path)
    torch.save(test_dataset , MODELS_DIR/ "test_data.pt")
    torch.save(val_dataset, MODELS_DIR/ "val_data.pt")
    torch.save( train_dataset, MODELS_DIR / "train_data.pt")
    logger.success(f"Test dataset saved to {test_data_path}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True  # Shuffle training data
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False  # No need to shuffle validation data
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False  # No need to shuffle test data
    )

    logger.success("Train, validation, and test datasets generation complete with labels.")
    return train_loader, val_loader, test_loader




def get_model_architecture(image_size, num_classes):
    """
    Return the model architecture used for training. (used also in predict.py)
    
    Args:
        image_size (int): The size of the input image (assumes square images).
        num_classes (int): The number of output classes.
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * (image_size // 4) * (image_size // 4), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_classes)
    )
    return model



# # Model training loop
def train_model(train_loader):

    """
    Summary of the function train_model.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        type: Description of the return value.
    """

    num_classes=len(train_loader.dataset.dataset.classes)
    # Define model architecture
    model = get_model_architecture(IMAGE_SIZE, num_classes)
    
    # Save number of classes for future use
    with open(MODELS_DIR / "num_classes.txt", "w") as f:
        f.write(str(num_classes))


    # Define optimizer (e.g., Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss function (e.g., CrossEntropyLoss)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()  # Set model to training mode
    for epoch in range(NUM_EPOCHS):  # NUM_EPOCHS is defined in config.py
        # Add a tqdm progress bar for the epoch
        progress_bar = tqdm(enumerate(train_loader), 
                            total=len(train_loader), 
                            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, (images, labels) in progress_bar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update progress bar with loss value
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] complete.")

        print("Training complete.")

    return model



@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    label_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    input_path: Path = PROCESSED_DATA_DIR,
    #output_path: Path = PROCESSED_DATA_DIR,
):
  
    logger.info(" Begining training  ")
    logger.info(" Loading training data ")
    train_loader, val_loader, test_loader = load_data(input_path)


    logger.info(" Training the model ")
    trained_model=train_model(train_loader)

    logger.info(f"Saving the trained model to {model_path}...")
    torch.save(trained_model.state_dict(), model_path)  # Save model weights
    logger.success(f"Model saved to {model_path}.")
    
    logger.success(" End training ")



if __name__ == "__main__":
    app()
