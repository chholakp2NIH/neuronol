import datetime
import os
from pathlib import Path

import mne
import pandas as pd
import pytest
from dotenv import load_dotenv

from neuronol.constants import (
    EASYCAP_EEG_CHANNELS,
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
    IMOTIONS_MARKERS_COL,
)
from neuronol.io.importer import DataImporter

# Given
load_dotenv()


@pytest.fixture
def fpath_eeg_csv():
    value = os.getenv("EEG_CSV_IMOTIONS")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_report():
    value = os.getenv("FPATH_MNE_REPORT")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_raw():
    value = os.getenv("FPATH_MNE_RAW")
    assert value is not None
    return value


@pytest.fixture
def model_events_sequence():
    fpath_events_seq = os.getenv("MODEL_EVENTS_SEQUENCE")
    assert fpath_events_seq is not None
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_restingstate_stimtimes():
    value = os.getenv("FPATH_RESTINGSTATE_STIMTIMES")
    assert value is not None
    return value


@pytest.fixture
def fpath_dig():
    value = os.getenv("FPATH_DIG")
    assert value is not None
    return value


@pytest.fixture
def fpath_eeg_csv_trigs():
    value = os.getenv("EEG_CSV_IMOTIONS_TRIGS")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_report_trigs():
    value = os.getenv("FPATH_MNE_REPORT_TRIGS")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_raw_trigs():
    value = os.getenv("FPATH_MNE_RAW_TRIGS")
    assert value is not None
    return value


@pytest.fixture
def model_events_sequence_trigs():
    fpath_events_seq = os.getenv("MODEL_EVENTS_SEQUENCE_TRIGS")
    assert fpath_events_seq is not None
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_eeg_csv_long():
    value = os.getenv("EEG_CSV_IMOTIONS_LONG")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_report_long():
    value = os.getenv("FPATH_MNE_REPORT_LONG")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_raw_long():
    value = os.getenv("FPATH_MNE_RAW_LONG")
    assert value is not None
    return value


@pytest.fixture
def model_events_sequence_long():
    fpath_events_seq = os.getenv("MODEL_EVENTS_SEQUENCE_LONG")
    assert fpath_events_seq is not None
    df_events = pd.read_csv(fpath_events_seq)
    return df_events["Event"]


@pytest.fixture
def fpath_headcircum():
    value = os.getenv("HEADCIRCUM_JSON")
    assert value is not None
    return value


# Run full data import (without embedded triggers)
def test_run_full(
    fpath_eeg_csv,
    fpath_mne_raw,
    fpath_mne_report,
    fpath_dig,
    fpath_restingstate_stimtimes,
):
    if Path(fpath_mne_raw).exists():
        Path(fpath_mne_raw).unlink()
    if Path(fpath_mne_report).exists():
        Path(fpath_mne_report).unlink()
    data_importer = DataImporter(
        fpath_eeg_csv,
        fpath_mne_raw=fpath_mne_raw,
        fpath_mne_report=fpath_mne_report,
        fpath_bs_dig=fpath_dig,
        event_files=[fpath_restingstate_stimtimes],
        gnd_channel="GND",
        renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    )
    data_importer.run()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert Path(fpath_mne_raw).exists()
    assert Path(fpath_mne_report).exists()
    assert IMOTIONS_BLINK_COL in data_importer.recording.raw.annotations.description
    assert "GND" in data_importer.recording.raw.ch_names
    assert data_importer.recording.raw.get_montage() == data_importer.recording.dig
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Run data import (with embedded triggers)
def test_run_with_embedded_triggers(
    fpath_eeg_csv_long,
):
    data_importer = DataImporter(
        fpath_eeg_csv_long,
    )
    data_importer.run()
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Run data import (without embedded triggers)
def test_run_without_embedded_triggers(
    fpath_eeg_csv,
    fpath_restingstate_stimtimes,
):
    # Test MNE Report and MNE Raw creation (in a file with embedded triggers)
    data_importer = DataImporter(
        fpath_eeg_csv, event_files=[fpath_restingstate_stimtimes]
    )
    data_importer.run()
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Create MNE raw from iMotions CSV
def test_create_mne_raw_from_imotions_csv(fpath_eeg_csv):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.create_mne_raw_from_imotions_csv()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)


# Read data
def test_read_imotions_data_as_df(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_data_as_df()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty


# Read iMotions preamble
def test_read_imotions_csv_preamble(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_preamble()
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read full iMotions CSV
def test_read_imotions_csv_full(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_full()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read recording date/time
def test_get_recording_datetime_from_imotions_preamble(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_preamble()
    data_importer.get_recording_datetime_from_imotions_preamble()
    assert isinstance(data_importer.recording.recording_dt, datetime.datetime)


# Get EEG data column numbers
def test_get_eeg_data_column_numbers(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_preamble()
    data_importer.get_eeg_data_column_numbers()
    assert all([isinstance(w, int) for w in data_importer.recording.eeg_col_nums])
    assert len(data_importer.recording.eeg_col_nums) > 0


# Extract EEG data from iMotions data
def test_extract_eeg_from_imotions_data(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data(eeg_channels=EASYCAP_EEG_CHANNELS)
    assert isinstance(data_importer.recording.df_eeg, pd.DataFrame)
    assert data_importer.recording.df_eeg.columns.tolist() == EASYCAP_EEG_CHANNELS
    assert not data_importer.recording.df_eeg.empty


# Extract head radius from dedicated JSON file
def test_head_radius(fpath_eeg_csv, fpath_headcircum):
    data_importer = DataImporter(fpath_eeg_csv, fpath_headcircum=fpath_headcircum)
    data_importer.evaluate_head_radius()
    assert isinstance(data_importer.recording.head_circum, float)
    assert isinstance(data_importer.recording.head_radius, float)
    assert data_importer.recording.head_circum < 1
    assert data_importer.recording.head_circum > 0


# Read ECG data from iMotions CSV and interpolate it to match EEG timepoints
def test_add_interpolated_ecg_to_eeg(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    assert data_importer.recording.df_eeg[IMOTIONS_ECG_COL].isna().sum() == 0


# Extract blink times from iMotions' data
def test_extract_event_times_from_imotions_data(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_full()
    data_importer.extract_event_times_from_imotions_data(
        IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
    )
    assert len(data_importer.recording.event_onsets) > 0
    assert (
        sum(
            [
                w == IMOTIONS_BLINK_COL
                for w in data_importer.recording.event_descriptions
            ]
        )
        > 0
    )


# Extract event triggers from iMotions' data
def test_read_event_markers_from_imotions_data(
    fpath_eeg_csv_trigs, model_events_sequence_trigs
):
    data_importer = DataImporter(fpath_eeg_csv_trigs)
    data_importer.read_imotions_csv_full()
    data_importer.read_event_markers_from_imotions_data()
    events_read_from_triggers = data_importer.recording.event_markers[
        IMOTIONS_MARKERS_COL
    ].values
    events_sequence_designed = model_events_sequence_trigs.values
    # n_events_min = min(len(events_read_from_triggers), len(events_sequence_designed))
    n_events = len(events_sequence_designed)
    assert all(events_read_from_triggers[:n_events] == events_sequence_designed)
    assert len(events_read_from_triggers) > 0
    assert len(data_importer.recording.event_onsets) > 0
    assert len(data_importer.recording.event_descriptions) > 0


# Create MNE Raw from extracted electrophys data
def test_convert_electrophys_data_to_mne_raw_object(
    fpath_eeg_csv,
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    data_importer.convert_electrophys_data_to_mne_raw_object()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert data_importer.recording.raw.ch_names == (
        EASYCAP_EEG_CHANNELS + [IMOTIONS_ECG_COL]
    )


# Read event markers from Excel and add to MNE Raw
def test_add_event_markers_from_event_files_to_mne_raw(
    fpath_eeg_csv, fpath_restingstate_stimtimes
):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.create_mne_raw_from_imotions_csv()
    data_importer.recording.raw.set_annotations(None)
    data_importer.add_event_markers_from_event_files_to_mne_raw(
        fpath_restingstate_stimtimes
    )
    assert "trials-start" in data_importer.recording.raw.annotations.description
    assert len(data_importer.recording.raw.annotations) > 0


#
def test_create_mne_montage_from_brainstorm_dig_data(fpath_eeg_csv, fpath_dig):
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.create_mne_montage_from_brainstorm_dig_data(fpath_dig)
    assert isinstance(data_importer.recording.dig, mne.channels.DigMontage)
    embedded_ch_names_in_dig = data_importer.recording.dig.ch_names
    data_importer = DataImporter(fpath_eeg_csv)
    data_importer.create_mne_montage_from_brainstorm_dig_data(
        fpath_dig, renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"]
    )
    renamed_ch_names_in_dig = data_importer.recording.dig.ch_names
    assert not (renamed_ch_names_in_dig == embedded_ch_names_in_dig)
    assert renamed_ch_names_in_dig == EASYCAP_EEG_CHANNELS + ["GND"]


# Log message
def test_log_message(fpath_eeg_csv, capsys):
    """
    Test log_message function
    """
    # Verbose: True
    data_importer = DataImporter(fpath_eeg_csv, verbose=True)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "\n!! Hello World! \n\n"
    # Verbose: False
    data_importer = DataImporter(fpath_eeg_csv, verbose=False)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == ""
