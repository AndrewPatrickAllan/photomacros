from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
# try:
#     from tqdm import tqdm

#     logger.remove(0)
#     logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
# except ModuleNotFoundError:
#     pass


# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


IMAGE_SIZE = 224  # Standard image size for models
MEAN = [0.485, 0.456, 0.406]  # Mean values for normalization
STD = [0.229, 0.224, 0.225]  # Standard deviation values for normalization
BATCH_SIZE = 32  # Number of samples in each batch
NUM_EPOCHS = 5 # Number of epochs to train the model should be minimum 5
test_data_path = PROCESSED_DATA_DIR / "test_dataset.pt"  # added ourselves, for predict.py to use in perform_inference function



# # ## quicker training test 
# IMAGE_SIZE = 40  # Standard image size for models
# BATCH_SIZE = 5  # Number of samples in each batch
# NUM_EPOCHS = 1 # Number of epochs to train the model should be minimum 5




