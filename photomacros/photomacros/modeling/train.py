from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    label_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = PROCESSED_DATA_DIR / "model.pkl",
    input_path: Path = PROCESSED_DATA_DIR,
    #output_path: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating train, val and test datasets...")
    image_paths = list(input_path.rglob("*.jpg"))
    dataset_size = len(image_paths)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)   # 15% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 15% for testing
    #randomly split datasets
    train_dataset, val_dataset, test_dataset = random_split(image_paths, [train_size, val_size, test_size])
    #put it into dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logger.success("train, val and test datasets generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
