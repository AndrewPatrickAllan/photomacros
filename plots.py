from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pickle 
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR,  NUM_EPOCHS#, initial_image_size, max_image_size, patience
import torchvision.models as models
import matplotlib.pyplot as plt



from photomacros.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # -----------------------------------------
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR ,
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

    history_path = Path(MODELS_DIR / f"HISTORY_model_{NUM_EPOCHS}epochs_init_LR_0P001_pretrainedDenseNet161_variable_LR_image_size.pkl")

    # Load the training history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch;'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch;'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(output_path / "epoch_vs_losses.png")
    logger.info(f"Plot saved to {output_path}")


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch;'], history['image_size'], label='Image Size')
    plt.xlabel('Epoch')
    plt.ylabel('iamge size')
    plt.legend()
    plt.savefig(output_path / "epoch_vs_image_size.png")
    logger.info(f"Plot saved to {output_path}")



    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch;'], history['top1_acc'], label='Top-1 Acc')
    plt.plot(history['epoch;'], history['top5_acc'], label='Top-5 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(output_path / "epoch_vs_acc.png")
    logger.info(f"Plot saved to {output_path}")


    logger.success("Plot generation complete.")
    # -----------------------------------------





if __name__ == "__main__":
    app()
