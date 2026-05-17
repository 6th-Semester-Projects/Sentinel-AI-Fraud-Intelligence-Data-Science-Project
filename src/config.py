import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data File Path
DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv.zip")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
    if not os.path.exists(DATA_FILE):
        DATA_FILE = os.path.join(BASE_DIR, "data", "raw", "creditcard_sample.csv")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
