import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter
from neuronol.io.watcher_noninter import DataWatcher
from neuronol.utilities import create_bids_file_paths_for_eegacamp_recording


def import_eegacamp_data(recording_data_dir):
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
        include_3d_plots=False,
        verbose=True,
    )
    data_importer.run()


# Given
data_dir = Path(os.path.expanduser("~/data/bids/imotions-sample/"))

# Create data watcher and start scouting
watcher = DataWatcher(import_eegacamp_data, data_dir)
watcher.scout()
