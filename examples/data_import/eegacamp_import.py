from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter
from neuronol.utilities import create_bids_fpaths_from_recording_dir

# Given
recording_data_dir = (
    # Path.home() / "data/bids/imotions-sample" / "sub-xx/ses-studyvisit2/eeg/"
    Path.home()
    / "data/bids/eegacamp-test"
    / "sub-006/ses-studyvisit3/eeg/"
)

# Read data and display
bids_fpaths = create_bids_fpaths_from_recording_dir(recording_data_dir)
data_importer = DataImporter(
    bids_fpaths["imotions-csv"],
    fpath_mne_raw=bids_fpaths["mne-raw"],
    fpath_mne_report=bids_fpaths["mne-report"],
    fpath_headcircum=bids_fpaths["headcircum"],
    fpath_bs_dig=bids_fpaths["headshape-dig"],
    event_files=bids_fpaths["events-stim-times-files"],
    gnd_channel="GND",
    renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    verbose=True,
)

# Run full data import
data_importer.run()
