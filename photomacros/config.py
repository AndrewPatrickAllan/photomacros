"""
Configuration file for setting paths, constants, and environment variables.

This script:
- Loads environment variables from a `.env` file.
- Defines paths for various directories and files in the project.
- Sets constants used for data processing and model training.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from a .env file, if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Define data directories
DATA_DIR = Path("~/Documents/GitHub/data").expanduser()  # Path to the main data directory
RAW_DATA_DIR = DATA_DIR / "raw"                         # Directory for raw, unprocessed data
INTERIM_DATA_DIR = DATA_DIR / "interim"                 # Directory for interim data during processing
PROCESSED_DATA_DIR = DATA_DIR / "processed"             # Directory for fully processed data
EXTERNAL_DATA_DIR = DATA_DIR / "external"               # Directory for external data sources
MODELS_DIR = PROJ_ROOT / "models"                       # Directory for saving and loading models
REPORTS_DIR = PROJ_ROOT / "reports"                     # Directory for reports
FIGURES_DIR = REPORTS_DIR / "figures"                   # Directory for report figures

# Image processing constants
IMAGE_SIZE = 224  # Image size used for training (quicker test configuration)
MEAN = [0.485, 0.456, 0.406]  # Mean values for image normalization
STD = [0.229, 0.224, 0.225]   # Standard deviation values for image normalization

# Model training constants
BATCH_SIZE = 32     # Number of samples in each batch (quicker test configuration)
NUM_EPOCHS = 3    # Number of epochs for training (quicker test configuration)

# Path for saved test data
#test_data_path = MODELS_DIR / "test_data.pt"  # Path to saved test dataset for inference