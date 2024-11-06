from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR ,
    output_path: Path = PROCESSED_DATA_DIR ,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
