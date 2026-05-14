from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
import numpy as np


# imported ourselves --------
import torch
from torch.utils.data import DataLoader
from photomacros.modeling.train import (
    get_model_architecture,
    get_validation_transforms,
)  # Importing own existing load_data function from train.py

# from torchvision import datasets, transforms
# import config
# from photomacros import dataset
# import random
# -------------------
# <<<<<<< HEAD
# from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR,  BATCH_SIZE,NUM_EPOCHS,MEAN,STD # IMAGE_SIZE,
# =======
from photomacros.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    initial_image_size,
    BATCH_SIZE,
    NUM_EPOCHS,
    FIGURES_DIR,
)

# >>>>>>> Checkdatasplit

IMAGE_SIZE = initial_image_size

# Set device globally
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


app = typer.Typer()


def load_model_into_eval_model(model_path: Path):
    """
    Load the trained model and set it to evaluation mode.

    Args:
        model_path (Path): Path to the trained model file (.pkl or .pth).

    Returns:
        model.eval(): Trained model set to evaluation mode.
    """

    # logger.info(f"Loading number of classes from {MODELS_DIR}/num_classes.txt...")
    # with open(MODELS_DIR / "num_classes.txt", "r") as f:
    #     num_classes = int(f.read().strip())
    num_classes = 101

    # Initialize the model
    logger.info("Initializing model architecture...")
    model = get_model_architecture(num_classes)
    print(f"Model architecture: {model}")

    # Load trained model
    logger.info(f"Loading trained model from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logger.success("Model loaded successfully.")

    return model


def perform_inference(
    model_path: Path, test_data_path: Path, predictions_path: Path, batch_size: int = 32
):
    """
    Perform inference on the test dataset using a trained model and save predictions to a file.

    Args:
        model_path (Path): Path to the trained model file (.pkl or .pth).
        test_data_path (Path): Path to the saved test dataset.
        predictions_path (Path): Path to save the predictions.
        batch_size (int): Batch size for DataLoader.
    """

    model = load_model_into_eval_model(model_path)
    model.to(device)

    # Load test dataset
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = torch.load(test_data_path)

    test_dataset.transform = get_validation_transforms(IMAGE_SIZE)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.success("Test dataset loaded successfully.")

    # Perform inference
    logger.info("Performing inference...")
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predicting"):
            images, labels = (
                images.to(device, dtype=torch.float32),
                labels.to(device),
            )  # Move data to GPU

            # print(f"Image batch shape: {images.shape}")
            outputs = model(images)
            print(f"Labels: {labels}")
            # print(f"Raw model outputs: {outputs[:5]}")  # Print first 5 predictions
            predicted_classes = outputs.argmax(dim=1)
            print(f"Predicted classes: {predicted_classes[:2]}")
            # Compare predictions with the labels
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted_classes.cpu().numpy())
            print(f"Predicted classes: {predictions}")

    accuracy = correct / total
    print(f"Accuracy for model_{NUM_EPOCHS}epochs.pkl : {accuracy:.4f}")

    # Save predictions
    logger.info(f"Saving predictions to {predictions_path}...")
    torch.save(predictions, predictions_path)  # Save as NumPy array
    logger.success(f"Predictions saved successfully to {predictions_path}.")


def save_test_labels(predictions, test_data_path: Path, output_path: Path):
    """
    Save predictions and corresponding labels to a file.

    Args:
        predictions (list): List of predicted labels.
        test_data_path (Path): Path to the test dataset file.
        output_path (Path): Path to save the labeled predictions.
    """
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = torch.load(test_data_path)
    test_dataset.dataset.transform = get_validation_transforms(IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Extracting ground truth labels from test_loader...")

    test_labels = []
    test_images = []

    # Iterate over test_loader to collect actual labels
    for batch_idx, (images, labels) in enumerate(test_loader):
        test_labels.extend(labels.cpu().numpy())  # Convert labels to list
        test_images.extend(
            [f"Image_{batch_idx * BATCH_SIZE + i}" for i in range(len(labels))]
        )  # Generate image names

    # # Ensure number of predictions and labels match
    # if len(predictions) != len(test_labels):
    #     logger.error("Mismatch between number of predictions and test labels!")
    #     raise ValueError("Mismatch between predictions and test labels.")

    logger.info("Making confusion matrix.")
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Generate confusion matrix
    y_true = np.array(test_labels)
    y_pred = np.array(predictions)

    # saving y_true and y_pred for later use
    np.save(REPORTS_DIR / "y_true.npy", y_true)
    np.save(REPORTS_DIR / "y_pred.npy", y_pred)

    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    print(
        f"y_true: {y_true[:5]}, y_pred: {y_pred[:5]}"
    )  # Print first 5 values for debugging
    print(y_pred[:5])

    # opening y_true and y_pred
    y_true = np.load(REPORTS_DIR / "y_true.npy")
    y_pred = np.load(REPORTS_DIR / "y_pred.npy")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(101))

    # cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm = cm_normalized

    print("CM::", np.round(cm_normalized[:5, :5], 2))

    # # Optionally normalize
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = test_dataset.dataset.classes

    cm_slice = cm[:30, :30]
    class_names_slice = class_names[:30]

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm_slice,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names_slice,
        yticklabels=class_names_slice,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (First 30 Classes)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.show()

    # # Display
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=test_dataset.dataset.classes)
    # fig, ax = plt.subplots(figsize=(20, 20))
    # disp.plot(ax=ax, cmap='viridis', xticks_rotation=90)
    # plt.show()

    # from sklearn.metrics import classification_report
    # import pandas as pd
    # import seaborn as sns

    # report = classification_report(y_true, y_pred, output_dict=True)
    # df_report = pd.DataFrame(report).transpose()

    # plt.figure(figsize=(18, 10))
    # sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='coolwarm')
    # plt.title("Classification Report Heatmap")
    # plt.show()

    # Create DataFrame
    test_df = pd.DataFrame(
        {
            "image": test_images,
            "ground_truth_label": test_labels,
            "predicted_label": predictions,
        }
    )

    # Save CSV
    logger.info(f"Saving labeled predictions to {output_path}...")
    test_df.to_csv(output_path, index=False)
    logger.success(f"Labeled predictions saved to {output_path}.")


@app.command()
def main(
    model_path: Path = MODELS_DIR
    / f"model_{NUM_EPOCHS}epochs_BetterModel_LR_Earlystop_pretrainedDenseNet_Overfit.pkl",
    predictions_path: Path = MODELS_DIR / "test_predictions.pt",
    test_data_path: Path = PROCESSED_DATA_DIR / "test_data.pt",
    test_labels_output_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
):
    """
    Main function to perform inference and save predictions with labels.
    """
    logger.info("Starting inference process...")
    perform_inference(
        model_path=model_path,
        test_data_path=test_data_path,
        predictions_path=predictions_path,
        batch_size=BATCH_SIZE,
    )

    # Load predictions
    predictions = torch.load(predictions_path)

    # Save predictions with corresponding labels
    logger.info(
        f"Saving predictions with corresponding labels to {test_labels_output_path}..."
    )
    save_test_labels(predictions, test_data_path, test_labels_output_path)
    logger.success("Inference process completed.")


if __name__ == "__main__":
    app()
