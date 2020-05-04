# Version library:  0.0.1
# =======================

import pathlib

import pandas as pd
import numpy as np
import config
import chardet
import glob

import joblib


# Individual pre-processing and training functions
# ================================================



# check the file encoding and load it into a pandas dataframe
# ===========================================================

def load_dataset(*, file_name: str) -> pd.DataFrame:

    """Load a persisted pipeline."""
    print("File".ljust(45), "Encoding")
    filename = f"{config.DATASET_DIR}/{file_name}"

    with open(filename, 'rb') as rawdata:
        result = chardet.detect(rawdata.read())
        print(file_name.ljust(45), result['encoding'])
        print('Cargando dataset...')

    _data = pd.read_csv(filename, encoding=result['encoding'])
    return _data

