import os
import time
from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter
from neuronol.io.watcher import DataWatcher
from neuronol.utilities import create_bids_fpaths_from_recording_dir


def import_eegacamp_data(recording_data_dir):
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
    data_importer.run()


# Given
data_dir = Path(
    os.environ.get(
        "EXAMPLE_DATA_DIR", os.path.expanduser("~/data/bids/imotions-sample/")
    )
)

# Create data watcher and start scouting
watcher = DataWatcher(data_dir)
watcher.observer.start()
try:
    while True:
        time.sleep(1)
        recording_data_dir = watcher.event_handler.queue.get()
        import_eegacamp_data(recording_data_dir)
finally:
    watcher.observer.stop()
    watcher.observer.join()
