import os
from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter

# Given
data_root = Path(os.path.expanduser("~/data/bids/imotions-sample/"))
fpath_import = data_root / Path("sub-xx/ses-studyvisit2/eeg/sub-xx_task-all_eeg.csv")
fpath_mne_raw = data_root / Path("sub-xx/ses-studyvisit2/eeg/sub-xx_task-all_eeg.fif")
fpath_mne_report = data_root / Path(
    "sub-xx/ses-studyvisit2/eeg/sub-xx_task-all_importreport.html"
)
fpath_headcircum = data_root / Path(
    "sub-xx/ses-studyvisit2/eeg/sub-xx_desc-manual_headcircumference.json"
)
fpath_dig = data_root / Path(
    "sub-xx/ses-studyvisit2/eeg/sub-xx_task-all_acq-polhemus_headshape.mat"
)
fpath_restingstate_stimtimes = data_root / Path(
    "sub-xx/ses-studyvisit2/eeg/sub-xx_task-restingstate_eventsstimtimes.xlsx"
)

# Read data and display
data_importer = DataImporter(
    fpath_import,
    fpath_mne_raw=fpath_mne_raw,
    fpath_mne_report=fpath_mne_report,
    fpath_headcircum=fpath_headcircum,
    fpath_bs_dig=fpath_dig,
    event_files=[fpath_restingstate_stimtimes],
    gnd_channel="GND",
    renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    verbose=True,
)

# Run full data import
data_importer.run()
