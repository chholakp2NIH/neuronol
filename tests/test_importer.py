import datetime
from pathlib import Path

import mne
import pandas as pd
import pytest

from neuronol.constants import (
    EASYCAP_EEG_CHANNELS,
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
    IMOTIONS_MARKERS_COL,
)
from neuronol.io.importer import DataImporter
from neuronol.utilities import create_bids_fpaths_from_recording_dir

"""
sub-xx: default shortest recording with blinks but no triggers
sub-x3: short recording with embedded triggers but no blinks
sub-x1: long recording with both blinks and embedded triggers
sub-yy: same iMotions recording as sub-xx but contains alternative Brainstorm dig
"""


# Fixtures
@pytest.fixture
def model_events_sequence():
    df_events = pd.read_csv(
        Path.home()
        / "data/bids/imotions-sample"
        / "derivatives/model_events_sequence_shortened.csv"
    )
    return df_events["Event"]


# Run full data import (without embedded triggers)
def test_run_full():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    if bids_fpaths["mne-raw"].exists():
        bids_fpaths["mne-raw"].unlink()
    if bids_fpaths["mne-report"].exists():
        bids_fpaths["mne-report"].unlink()
    data_importer = DataImporter(
        bids_fpaths["imotions-csv"],
        fpath_mne_raw=bids_fpaths["mne-raw"],
        fpath_mne_report=bids_fpaths["mne-report"],
        fpath_bs_dig=bids_fpaths["headshape-dig"],
        event_files=bids_fpaths["events-stim-times-files"],
        gnd_channel="GND",
        renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    )
    data_importer.run()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert bids_fpaths["mne-raw"].exists()
    assert bids_fpaths["mne-report"].exists()
    assert IMOTIONS_BLINK_COL in data_importer.recording.raw.annotations.description
    assert "GND" in data_importer.recording.raw.ch_names
    assert data_importer.recording.raw.get_montage() == data_importer.recording.dig
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Run data import (with embedded triggers)
def test_run_with_embedded_triggers():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-x3" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.run()
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Run data import (without embedded triggers)
def test_run_without_embedded_triggers():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    # Test MNE Report and MNE Raw creation (using a file with triggers)
    data_importer = DataImporter(
        bids_fpaths["imotions-csv"],
        event_files=bids_fpaths["events-stim-times-files"],
    )
    data_importer.run()
    assert "trials-start" in data_importer.recording.raw.annotations.description


# Create MNE raw from iMotions CSV
def test_create_mne_raw_from_imotions_csv():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.create_mne_raw_from_imotions_csv()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)


# Read data
def test_read_imotions_data_as_df():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_data_as_df()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty


# Read iMotions preamble
def test_read_imotions_csv_preamble():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_preamble()
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read full iMotions CSV
def test_read_imotions_csv_full():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    assert isinstance(data_importer.recording.df_raw, pd.DataFrame)
    assert not data_importer.recording.df_raw.empty
    assert isinstance(data_importer.recording.preamble, str)
    assert len(data_importer.recording.preamble) > 0


# Read recording date/time
def test_get_recording_datetime_from_imotions_preamble():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_preamble()
    data_importer.get_recording_datetime_from_imotions_preamble()
    assert isinstance(data_importer.recording.recording_dt, datetime.datetime)


# Get EEG data column numbers
def test_get_eeg_data_column_numbers():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_preamble()
    data_importer.get_eeg_data_column_numbers()
    assert all([isinstance(w, int) for w in data_importer.recording.eeg_col_nums])
    assert len(data_importer.recording.eeg_col_nums) > 0


# Extract EEG data from iMotions data
def test_extract_eeg_from_imotions_data():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data(eeg_channels=EASYCAP_EEG_CHANNELS)
    assert isinstance(data_importer.recording.df_eeg, pd.DataFrame)
    assert data_importer.recording.df_eeg.columns.tolist() == EASYCAP_EEG_CHANNELS
    assert not data_importer.recording.df_eeg.empty


# Extract head radius from dedicated JSON file
def test_head_radius():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(
        bids_fpaths["imotions-csv"], fpath_headcircum=bids_fpaths["headcircum"]
    )
    data_importer.evaluate_head_radius()
    assert isinstance(data_importer.recording.head_circum, float)
    assert isinstance(data_importer.recording.head_radius, float)
    assert data_importer.recording.head_circum < 1
    assert data_importer.recording.head_circum > 0


# Read ECG data from iMotions CSV and interpolate it to match EEG timepoints
def test_add_interpolated_ecg_to_eeg():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    assert data_importer.recording.df_eeg[IMOTIONS_ECG_COL].isna().sum() == 0


# Extract blink times from iMotions' data
def test_extract_event_times_from_imotions_data():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    data_importer.extract_event_times_from_imotions_data(
        IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
    )  # iMotions CSV with blinks present
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
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-yy" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    with pytest.raises(
        ValueError, match=f"No positive instance found for event: {IMOTIONS_BLINK_COL}"
    ):
        data_importer.extract_event_times_from_imotions_data(
            IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
        )


# Extract event triggers from iMotions' data
def test_read_event_markers_from_imotions_data(
    model_events_sequence,
):
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-x3" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    data_importer.read_event_markers_from_imotions_data()
    events_read_from_triggers = data_importer.recording.event_markers[
        IMOTIONS_MARKERS_COL
    ].values
    events_sequence_designed = model_events_sequence.values
    n_events = len(events_sequence_designed)
    assert all(events_read_from_triggers[:n_events] == events_sequence_designed)
    assert len(events_read_from_triggers) > 0
    assert len(data_importer.recording.event_onsets) > 0
    assert len(data_importer.recording.event_descriptions) > 0


# Create MNE Raw from extracted electrophys data
def test_convert_electrophys_data_to_mne_raw_object():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.read_imotions_csv_full()
    data_importer.extract_eeg_from_imotions_data()
    data_importer.add_interpolated_ecg_to_eeg()
    data_importer.convert_electrophys_data_to_mne_raw_object()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert data_importer.recording.raw.ch_names == (
        EASYCAP_EEG_CHANNELS + [IMOTIONS_ECG_COL]
    )


# Read event markers from Excel and add to MNE Raw
def test_add_event_markers_from_event_files_to_mne_raw():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer = DataImporter(bids_fpaths["imotions-csv"])
    data_importer.create_mne_raw_from_imotions_csv()
    data_importer.recording.raw.set_annotations(None)
    for fpath in bids_fpaths["events-stim-times-files"]:
        data_importer.add_event_markers_from_event_files_to_mne_raw(fpath)
    assert "trials-start" in data_importer.recording.raw.annotations.description
    assert len(data_importer.recording.raw.annotations) > 0


# Create MNE dig montage from Brainstorm dig data
def test_create_mne_montage_from_brainstorm_dig_data(
    tmp_path,
):
    # Default import
    data_importer = DataImporter(tmp_path)
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-xx" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer.create_mne_montage_from_brainstorm_dig_data(
        bids_fpaths["headshape-dig"]
    )
    assert isinstance(data_importer.recording.dig, mne.channels.DigMontage)
    # Import with channels renamed
    embedded_ch_names_in_dig = data_importer.recording.dig.ch_names
    data_importer = DataImporter(tmp_path)
    data_importer.create_mne_montage_from_brainstorm_dig_data(
        bids_fpaths["headshape-dig"],
        renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    )
    renamed_ch_names_in_dig = data_importer.recording.dig.ch_names
    assert not (renamed_ch_names_in_dig == embedded_ch_names_in_dig)
    assert renamed_ch_names_in_dig == EASYCAP_EEG_CHANNELS + ["GND"]
    # Import with alternative nasion labels in dig data
    data_importer = DataImporter(tmp_path)
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/imotions-sample" / "sub-yy" / "ses-studyvisit2/eeg"
    )  # iMotions CSV with no blinks
    data_importer.create_mne_montage_from_brainstorm_dig_data(
        bids_fpaths["headshape-dig"]
    )
    assert isinstance(data_importer.recording.dig, mne.channels.DigMontage)


# Log message
def test_log_message(tmp_path, capsys):
    """
    Test log_message function
    """
    # Verbose: True
    data_importer = DataImporter(tmp_path, verbose=True)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "\n!! Hello World! \n\n"
    # Verbose: False
    data_importer = DataImporter(tmp_path, verbose=False)
    data_importer._log_message("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == ""
