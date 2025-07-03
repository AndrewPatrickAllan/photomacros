
from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
import numpy as np


# imported ourselves --------
import torch
from torch.utils.data import DataLoader
from train import load_data, get_model_architecture,get_validation_transforms # Importing own existing load_data function from train.py
# from torchvision import datasets, transforms
# import config
# from photomacros import dataset
# import random
# -------------------
# <<<<<<< HEAD
# from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR,  BATCH_SIZE,NUM_EPOCHS,MEAN,STD # IMAGE_SIZE,
# =======
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR,initial_image_size, BATCH_SIZE,NUM_EPOCHS,MEAN,STD, FIGURES_DIR
# >>>>>>> Checkdatasplit
from torchvision import datasets, transforms
IMAGE_SIZE=initial_image_size


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np




test_data_path: Path = MODELS_DIR/ "test_data.pt",

test_dataset=torch.load(test_data_path)
test_dataset.dataset.transform = get_validation_transforms(IMAGE_SIZE)

# opening y_true and y_pred
y_true = np.load(MODELS_DIR / "y_true.npy")
y_pred = np.load(MODELS_DIR / "y_pred.npy")

cm = confusion_matrix(y_true, y_pred, labels=np.arange(101))

# Optionally normalize
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=test_dataset.dataset.classes)
fig, ax = plt.subplots(figsize=(20, 20))  
disp.plot(ax=ax, cmap='viridis', xticks_rotation=90)
plt.title("Normalized Confusion Matrix")
plt.savefig(FIGURES_DIR  / "confusion_matrix.png")
plt.show()