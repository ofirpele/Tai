import os
from pathlib import Path

RAW_DATA_DIR_PATH = (
    os.path.join(Path(__file__).resolve().parents[0], "data", "raw_data") + "\\"
)
MERGED_DATA_DIR_PATH = (
    os.path.join(Path(RAW_DATA_DIR_PATH).resolve().parents[0], "merged_data") + "\\"
)
MERGED_DATA_FILE =  MERGED_DATA_DIR_PATH + 'data.csv'
