from pathlib import Path
from PIL import Image
import typer
from loguru import logger
from tqdm import tqdm
import os 

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_dir: Path =  RAW_DATA_DIR /"food101/images/",   # Directory with raw images
    output_dir: Path = PROCESSED_DATA_DIR # Directory to save processed images

):
    """
    Process all .jpg images from the input directory and save to the output directory.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .jpg files in the input directory
    image_paths = list(input_dir.rglob("*.jpg"))
    print (input_dir.glob("*.jpg"))

    logger.info(f"Found {len(image_paths)} .jpg files to process.")

    # Loop through each image in the input directory
    for img_path in tqdm(image_paths, total=len(image_paths), desc="Processing images"):
        try:
            # Open the image
            with Image.open(img_path) as img:
                # Example: Resize the image to 256x256 and save it
                img_resized = img.resize((256, 256))
                
                # Create a new file path in the output directory, maintaining the subdirectory structure after 'images/'
                relative_path = img_path.relative_to(input_dir)  # Get the relative path from input_dir
                output_image_path = output_dir / relative_path  # Append it to the output directory path

                # Ensure the output subdirectory exists
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the resized image
                img_resized.save(output_image_path)
                logger.info(f"Processed and saved {output_image_path}")
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")

    logger.success("All images processed successfully.")

if __name__ == "__main__":
    app()