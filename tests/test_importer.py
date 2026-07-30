import datetime
import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from neuronol.constants import (
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
)
from neuronol.io.importer import DataImporter

# import sys
# from pathlib import Path
# # Add the outer parent directory to the path list
# # This corresponds to: /Users/chholakp2/analysis/neuronol
# project_root = Path(__file__).resolve().parent.parent
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))


# Given
@pytest.fixture
def fpath_eeg_csv():
    load_dotenv()
    return os.getenv("EEG_CSV_IMOTIONS")


@pytest.fixture
def fpath_headcircum():
    load_dotenv()
    return os.getenv("HEADCIRCUM_JSON")


# Read data
def test_read_imotions_data_as_df(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_data_as_df()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty


# Read iMotions preamble
def test_read_imotions_csv_preamble(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_preamble()
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read full iMotions CSV
def test_read_imotions_csv_full(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read recording date/time
def test_get_recording_datetime_from_imotions_preamble(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_preamble()
    data_importer.get_recording_datetime_from_imotions_preamble()
    assert isinstance(data_importer.recording.recording_dt, datetime.datetime)


# Get EEG data column numbers
def test_get_eeg_data_column_numbers(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_preamble()
    data_importer.get_eeg_data_column_numbers()
    assert all([isinstance(w, int) for w in data_importer.recording.eeg_col_nums])
    assert len(data_importer.recording.eeg_col_nums) > 0


# Extract EEG data from iMotions data
def test_extract_eeg_from_imotions_data(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    assert isinstance(data_importer.recording.df_eeg, pd.DataFrame)
    assert not data_importer.recording.df_eeg.empty


# Extract head radius from dedicated JSON file
def test_head_radius(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.evaluate_head_radius()
    assert isinstance(data_importer.recording.head_circum, float)
    assert isinstance(data_importer.recording.head_radius, float)
    assert data_importer.recording.head_circum < 1
    assert data_importer.recording.head_circum > 0


# Read ECG data from iMotions CSV and interpolate it to match EEG timepoints
def test_add_interpolated_ecg_to_eeg(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    assert data_importer.recording.df_eeg[IMOTIONS_ECG_COL].isna().sum() == 0


# Extract blink times from iMotions' data
def test_extract_event_times_from_imotions_data(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    times = data_importer.extract_event_times_from_imotions_data(
        IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
    )
    assert len(times) > 0


# Log message
def test_log_message(fpath_eeg_csv, fpath_headcircum, capsys):
    """
    Test log_message function
    """
    # Verbose: True
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum, verbose=True)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "\n!! Hello World! \n\n"
    # Verbose: False
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum, verbose=False)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == ""
