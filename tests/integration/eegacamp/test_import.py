import os
from pathlib import Path

import mne
import pytest
from dotenv import load_dotenv

from neuronol.constants import EASYCAP_EEG_CHANNELS, IMOTIONS_BLINK_COL
from neuronol.io.importer import DataImporter

load_dotenv(str(Path(__file__).resolve().parent / ".env"))


@pytest.fixture
def eegacamp_data_dir():
    value = os.getenv("EEGACAMP_DATA_DIR")
    assert value is not None
    return value


@pytest.fixture
def fpaths_rec(eegacamp_data_dir):
    data_dir = Path(eegacamp_data_dir)
    sub_id = 1
    recording_data_dir = data_dir / f"sub-{sub_id:03d}/ses-studyvisit2/eeg/"
    fpaths = {
        "fpath_import": recording_data_dir / f"sub-{sub_id:03d}_task-all_eeg.csv",
        "fpath_mne_raw": recording_data_dir / f"sub-{sub_id:03d}_task-all_eeg.fif",
        "fpath_mne_report": (
            recording_data_dir / f"sub-{sub_id:03d}_task-all_importreport.html"
        ),
        "fpath_headcircum": (
            recording_data_dir / f"sub-{sub_id:03d}_desc-manual_headcircumference.json"
        ),
        "fpath_bs_dig": (
            recording_data_dir / f"sub-{sub_id:03d}_task-all_acq-polhemus_headshape.mat"
        ),
        "event_files": recording_data_dir.glob("*_eventsstimtimes.xlsx"),
    }
    return fpaths


# @pytest.mark.integration
def test_eegacamp_import(fpaths_rec):
    if fpaths_rec["fpath_mne_raw"].exists():
        fpaths_rec["fpath_mne_raw"].unlink()
    if fpaths_rec["fpath_mne_report"].exists():
        fpaths_rec["fpath_mne_report"].unlink()
    data_importer = DataImporter(
        fpaths_rec["fpath_import"],
        fpath_mne_raw=fpaths_rec["fpath_mne_raw"],
        fpath_mne_report=fpaths_rec["fpath_mne_report"],
        fpath_headcircum=fpaths_rec["fpath_headcircum"],
        fpath_bs_dig=fpaths_rec["fpath_bs_dig"],
        event_files=fpaths_rec["event_files"],
        gnd_channel="GND",
        renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    )
    data_importer.run()
    assert isinstance(data_importer.recording.raw, mne.io.RawArray)
    assert fpaths_rec["fpath_mne_raw"].exists()
    assert fpaths_rec["fpath_mne_report"].exists()
    assert IMOTIONS_BLINK_COL in data_importer.recording.raw.annotations.description
    assert "GND" in data_importer.recording.raw.ch_names
    assert data_importer.recording.raw.get_montage() == data_importer.recording.dig
    assert "trials-start" in data_importer.recording.raw.annotations.description
