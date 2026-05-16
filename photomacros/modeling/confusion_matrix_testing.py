from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# imported ourselves --------
import torch
from photomacros.modeling.train import (
    get_validation_transforms,
)  # Importing own existing load_data function from train.py

from photomacros.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    initial_image_size,
    FIGURES_DIR,
)

IMAGE_SIZE = initial_image_size


test_data_path: Path = (PROCESSED_DATA_DIR / "test_data.pt",)

test_dataset = torch.load(test_data_path)
test_dataset.dataset.transform = get_validation_transforms(IMAGE_SIZE)

# opening y_true and y_pred
y_true = np.load(REPORTS_DIR / "y_true.npy")
y_pred = np.load(REPORTS_DIR / "y_pred.npy")

cm = confusion_matrix(y_true, y_pred, labels=np.arange(101))

# Optionally normalize
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized, display_labels=test_dataset.dataset.classes
)
fig, ax = plt.subplots(figsize=(20, 20))
disp.plot(ax=ax, cmap="viridis", xticks_rotation=90)
plt.title("Normalized Confusion Matrix")
plt.savefig(FIGURES_DIR / "confusion_matrix.png")
plt.show()
