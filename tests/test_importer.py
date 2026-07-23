import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

# Add the outer parent directory to the path list
# This corresponds to: /Users/chholakp2/analysis/neuronol
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neuronol.io.importer import DataImporter


# Given
@pytest.fixture
def fpath_eeg_csv():
    load_dotenv()
    return os.getenv("EEG_CSV_IMOTIONS")


# Read data
def test_read_imotions_csv_as_df(fpath_eeg_csv):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_as_df()
    assert isinstance(data_importer.df_raw, pd.DataFrame)
    assert not data_importer.df_raw.empty


# Read recording date/time
def test_get_recording_datetime_from_imotions_df(fpath_eeg_csv):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_as_df()
    data_importer.get_recording_datetime_from_imotions_df()
    assert isinstance(data_importer.recording_dt, datetime.datetime)


# Drop header info from imotions df
def test_drop_header_info_from_df(fpath_eeg_csv):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_as_df()
    data_importer.drop_header_info_from_df()
    assert "Timestamp" in data_importer.df_raw.columns.to_list()
