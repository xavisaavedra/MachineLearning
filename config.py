
import pathlib

# ====   PATHS ===================

# PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent

PACKAGE_ROOT = pathlib.Path('.').resolve()
DATASET_DIR = PACKAGE_ROOT / "DATA"

OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'model.pkl'

# data
TESTING_DATA_FILE = ""
TRAINING_DATA_FILE = "train.csv"
TARGET = "survived"

# ======= PARAMETERS ===============

# imputation parameters


# encoding parameters


# ======= FEATURE GROUPS =============
                     
# variable groups for engineering steps


# variables to transofmr


# variables to encode


# selected features for training

