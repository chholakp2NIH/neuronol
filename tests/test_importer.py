import datetime
import os
from pathlib import Path

import mne
import pandas as pd
import pytest
from dotenv import load_dotenv
from numpy import isin

from neuronol.constants import (
    EASYCAP_EEG_CHANNELS,
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
    IMOTIONS_MARKERS_COL,
)
from neuronol.io.importer import DataImporter


# Given
@pytest.fixture
def fpath_eeg_csv():
    load_dotenv()
    return os.getenv("EEG_CSV_IMOTIONS")


@pytest.fixture
def fpath_mne_report():
    load_dotenv()
    return os.getenv("FPATH_MNE_REPORT")


@pytest.fixture
def fpath_mne_raw():
    load_dotenv()
    return os.getenv("FPATH_MNE_RAW")


@pytest.fixture
def model_events_sequence():
    load_dotenv()
    fpath_events_seq: str = os.getenv("MODEL_EVENTS_SEQUENCE")
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_eeg_csv_trigs():
    load_dotenv()
    return os.getenv("EEG_CSV_IMOTIONS_TRIGS")


@pytest.fixture
def fpath_mne_report_trigs():
    load_dotenv()
    return os.getenv("FPATH_MNE_REPORT_TRIGS")


@pytest.fixture
def fpath_mne_raw_trigs():
    load_dotenv()
    return os.getenv("FPATH_MNE_RAW_TRIGS")


@pytest.fixture
def model_events_sequence_trigs():
    load_dotenv()
    fpath_events_seq: str = os.getenv("MODEL_EVENTS_SEQUENCE_TRIGS")
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_eeg_csv_long():
    load_dotenv()
    return os.getenv("EEG_CSV_IMOTIONS_LONG")


@pytest.fixture
def fpath_mne_report_long():
    load_dotenv()
    return os.getenv("FPATH_MNE_REPORT_LONG")


@pytest.fixture
def fpath_mne_raw_long():
    load_dotenv()
    return os.getenv("FPATH_MNE_RAW_LONG")


@pytest.fixture
def model_events_sequence_long():
    load_dotenv()
    fpath_events_seq: str = os.getenv("MODEL_EVENTS_SEQUENCE_LONG")
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_headcircum():
    load_dotenv()
    return os.getenv("HEADCIRCUM_JSON")


# Run full data import
def test_run(
    fpath_eeg_csv_long, fpath_headcircum, fpath_mne_raw_long, fpath_mne_report_long
):
    data_importer = DataImporter(
        fpath_eeg_csv_long, fpath_headcircum, fpath_mne_report=fpath_mne_report_long
    )
    data_importer.run()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert Path(fpath_mne_report_long).exists()
    data_importer = DataImporter(
        fpath_eeg_csv_long, fpath_headcircum, fpath_mne_raw=fpath_mne_raw_long
    )
    data_importer.run()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert Path(fpath_mne_raw_long).exists()


# Create MNE raw from iMotions CSV
def test_create_mne_raw_from_imotions_csv(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.create_mne_raw_from_imotions_csv()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)


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
    data_importer.extract_eeg_from_imotions_data(eeg_channels=EASYCAP_EEG_CHANNELS)
    assert isinstance(data_importer.recording.df_eeg, pd.DataFrame)
    assert data_importer.recording.df_eeg.columns.tolist() == EASYCAP_EEG_CHANNELS
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
    data_importer.extract_event_times_from_imotions_data(
        IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
    )
    assert len(data_importer.recording.blink_times) > 0
    assert len(data_importer.recording.event_onsets) > 0
    assert sum([w == "blink" for w in data_importer.recording.event_descriptions]) > 0


# Extract event triggers from iMotions' data
def test_read_event_markers_from_imotions_data(
    fpath_eeg_csv_trigs, fpath_headcircum, model_events_sequence_trigs
):
    data_importer = DataImporter(fpath_eeg_csv_trigs, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    data_importer.read_event_markers_from_imotions_data()
    events_read_from_triggers = data_importer.recording.event_markers[
        IMOTIONS_MARKERS_COL
    ].values
    events_sequence_designed = model_events_sequence_trigs.values
    n_events_min = min(len(events_read_from_triggers), len(events_sequence_designed))
    assert all(
        events_read_from_triggers[:n_events_min]
        == events_sequence_designed[:n_events_min]
    )
    assert len(events_read_from_triggers) > 0
    assert len(data_importer.recording.event_onsets) > 0
    assert len(data_importer.recording.event_descriptions) > 0


# Create MNE Raw from extracted electrophys data
def test_convert_electrophys_data_to_mne_raw_object(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    data_importer.convert_electrophys_data_to_mne_raw_object()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert data_importer.recording.raw.ch_names == (
        EASYCAP_EEG_CHANNELS + [IMOTIONS_ECG_COL]
    )


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
