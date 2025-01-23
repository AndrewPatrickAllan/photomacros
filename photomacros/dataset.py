"""
Image Processing Script

This script processes `.jpg` images from a specified input directory, resizes them to 256x256, 
and saves them in the specified output directory while preserving the original subdirectory structure.

Features:
- Recursively processes images in the input directory.
- Ensures the output directory structure matches the input.
- Logs processing progress and errors using `loguru` and `tqdm`.
"""

from pathlib import Path
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
import os

from photomacros.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR / "food101/images/",  # Default input directory containing raw images
    output_dir: Path = PROCESSED_DATA_DIR               # Default output directory for processed images
):
    """
    Resize and process all .jpg images from the input directory and save to the output directory.

    Parameters
    ----------
    input_dir : Path, optional
        Path to the directory containing raw images. Defaults to Path/"food101/images".
    output_dir : Path, optional
        Path to the directory where processed images will be saved. Defaults to Path/processed_data_dir.

    Returns
    -------
    None
        This function does not return anything. It processes images and saves them to the output directory.
    """
    # Ensure the output directory exists, creating it if necessary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively find all .jpg files in the input directory
    image_paths = list(input_dir.rglob("*.jpg"))  # Finds all .jpg files in input_dir and its subdirectories
    logger.info(f"Found {len(image_paths)} .jpg files to process.")

    # Iterate through each image path
    for img_path in tqdm(image_paths, total=len(image_paths), desc="Processing images"):
        try:
            # Open the image using the Pillow (PIL) library
            with Image.open(img_path) as img:
                # Resize the image to a fixed size of 256x256 pixels
                img_resized = img.resize((256, 256))
                
                # Preserve the original subdirectory structure after the input directory
                relative_path = img_path.relative_to(input_dir)  # Get the relative path of the image
                output_image_path = output_dir / relative_path  # Combine it with the output directory

                # Ensure the subdirectory structure in the output directory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the resized image to the output directory
                img_resized.save(output_image_path)
                logger.info(f"Processed and saved {output_image_path}")
        except Exception as e:
            # Log any errors that occur during processing
            logger.error(f"Failed to process {img_path}: {e}")

    # Log a success message when all images have been processed
    logger.success("All images processed successfully.")

if __name__ == "__main__":
    app()