import os

# from datetime import datetime
from pathlib import Path

import mne
from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, add_recording_to_db

# from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.dbmanager import DBManager

# from neuronol.io.importer import DataImporter
from neuronol.utilities import (
    create_bids_fpaths_from_recording_dir,
)  # get_sub_and_ses_from_bids_rec_dir,

# Given
data_dir = Path(
    os.environ.get(
        "EXAMPLE_DATA_DIR",
        os.path.expanduser("~/data/bids/eegacamp/"),
    )
)
recording_data_dir = data_dir / "sub-002/ses-studyvisit2/eeg/"
# fpath_db = data_dir / "derivatives/imaging.db"
# fpath_db = Path.home() / "data/bids/imotions-sample" / "derivatives/imaging.db"
fnames_csv = [
    "sub-002_task-all_run-1_eeg.csv",
    "sub-002_task-all_run-2_eeg.csv",
]
fname_raw_concat = "sub-002_task-all_eeg.fif"

# Run data importer on all runs
fpaths_mne_raw = []
for fname_csv in fnames_csv:
    bids_fpaths = create_bids_fpaths_from_recording_dir(recording_data_dir)
    fpath_import = recording_data_dir / fname_csv
    fpath_mne_raw = fpath_import.with_suffix(".fif")
    fpaths_mne_raw.append(fpath_mne_raw)
    # data_importer = DataImporter(
    #     fpath_import,
    #     fpath_mne_raw=fpath_mne_raw,
    #     fpath_mne_report=fpath_import.with_name(
    #         fpath_import.stem.removesuffix("_eeg") + "_importreport.html"
    #     ),
    #     fpath_headcircum=bids_fpaths["headcircum"],
    #     fpath_bs_dig=bids_fpaths["headshape-dig"],
    #     event_files=bids_fpaths["events-stim-times-files"],
    #     gnd_channel="GND",
    #     renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
    #     verbose=True,
    # )
    # # Run full data import
    # data_importer.run()

# Combine imported data into 1 FIF file
fpath_raw_concat = recording_data_dir / fname_raw_concat
raws = [mne.io.read_raw_fif(fpath) for fpath in fpaths_mne_raw]
raw_concat = mne.concatenate_raws(raws, preload=True)
# raw_concat.save(fpath_raw_concat, overwrite=True)
