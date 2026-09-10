from pathlib import Path

import mne

from neuronol.constants import EASYCAP_EEG_CHANNELS, IMOTIONS_BLINK_COL
from neuronol.io.importer import DataImporter
from neuronol.utilities import create_bids_fpaths_from_recording_dir


# @pytest.mark.integration
def test_eegacamp_import():
    bids_fpaths = create_bids_fpaths_from_recording_dir(
        Path.home() / "data/bids/eegacamp-test" / "sub-001" / "ses-studyvisit2/eeg"
    )
    if bids_fpaths["mne-raw"].exists():
        bids_fpaths["mne-raw"].unlink()
    if bids_fpaths["mne-report"].exists():
        bids_fpaths["mne-report"].unlink()
    data_importer = DataImporter(
        bids_fpaths["imotions-csv"],
        fpath_mne_raw=bids_fpaths["mne-raw"],
        fpath_mne_report=bids_fpaths["mne-report"],
        fpath_headcircum=bids_fpaths["headcircum"],
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
