from pathlib import Path

import typer
from loguru import logger
import pickle
from photomacros.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    NUM_EPOCHS,
)  # , initial_image_size, max_image_size, patience
import matplotlib.pyplot as plt


from photomacros.config import FIGURES_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    """
    Generating plot of accuracies over epochs

    Parameters
    ----------
    input_dir : Path
    output_dir : Path

    Returns
    -------
    None
    """
    # -----------------------------------------
    logger.info("Generating plot of accuracies over epochs.")

    history_path = Path(
        MODELS_DIR
        / f"HISTORY_model_{NUM_EPOCHS}epochs_init_LR_0P001_pretrainedDenseNet161_variable_LR_image_size.pkl"
    )

    # Load the training history
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
