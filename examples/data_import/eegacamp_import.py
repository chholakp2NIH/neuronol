from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter
from neuronol.utilities import create_bids_file_paths_for_eegacamp_recording

# Given
recording_data_dir = (
    Path.home() / "data/bids/imotions-sample" / "sub-xx/ses-studyvisit2/eeg/"
)

# Read data and display
bids_fpaths = create_bids_file_paths_for_eegacamp_recording(recording_data_dir)
data_importer = DataImporter(
    bids_fpaths["fpath_import"],
    fpath_mne_raw=bids_fpaths["fpath_mne_raw"],
    fpath_mne_report=bids_fpaths["fpath_mne_report"],
    fpath_headcircum=bids_fpaths["fpath_headcircum"],
    fpath_bs_dig=bids_fpaths["fpath_bs_dig"],
    event_files=bids_fpaths["event_files"],
    gnd_channel="GND",
    renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    verbose=True,
)

# Run full data import
data_importer.run()
