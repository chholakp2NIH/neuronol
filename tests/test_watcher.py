from pathlib import Path

from watchdog.events import FileSystemEvent
from watchdog.observers.api import BaseObserver

from neuronol.constants import EASYCAP_EEG_CHANNELS, WATCHER_DATA_COPY_COMPLETED
from neuronol.io.importer import DataImporter
from neuronol.io.watcher import DataEventHandler, DataWatcher
from neuronol.utilities import create_bids_fpaths_from_recording_dir


def test_handler_init():
    handler = DataEventHandler()
    assert handler.queue.empty()


def test_handler_on_created():
    handler = DataEventHandler()
    testing_paths = {
        "Directory": ".",
        "UnrelatedFile": "./xyz",
        "CorrectFile": f"./{WATCHER_DATA_COPY_COMPLETED}",
    }
    for case_path in testing_paths:
        event = FileSystemEvent(testing_paths[case_path])
        handler.on_created(event)
        if case_path == "Directory":
            assert handler.queue.empty()
        elif case_path == "UnrelatedFile":
            assert handler.queue.empty()
        elif case_path == "CorrectFile":
            assert handler.queue.get() == Path(testing_paths["CorrectFile"]).parent


def test_observer_init(tmp_path):
    watcher = DataWatcher(tmp_path)
    assert isinstance(watcher.observer, BaseObserver)
    assert isinstance(watcher.event_handler, DataEventHandler)


def test_example_integration():
    data_dir = Path.home() / "data/bids/imotions-sample"
    recording_data_dir_orig = data_dir / "sub-xx/ses-studyvisit2/eeg"
    fpath_data_completion_flag = recording_data_dir_orig / WATCHER_DATA_COPY_COMPLETED
    bids_fpaths_orig = create_bids_fpaths_from_recording_dir(recording_data_dir_orig)
    # Set up watcher
    watcher = DataWatcher(data_dir)
    watcher.observer.start()
    # Trigger watcher
    if bids_fpaths_orig["mne-raw"].exists():
        bids_fpaths_orig["mne-raw"].unlink()
    if bids_fpaths_orig["mne-report"].exists():
        bids_fpaths_orig["mne-report"].unlink()
    if fpath_data_completion_flag.exists():
        fpath_data_completion_flag.unlink()
    fpath_data_completion_flag.touch()
    # Run data import once watcher is triggered
    recording_data_dir = watcher.event_handler.queue.get()
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
    # Stop observer once data reading completed
    watcher.observer.stop()
    watcher.observer.join()
    # Assertions
    assert bids_fpaths_orig["mne-raw"].exists()
    assert bids_fpaths_orig["mne-report"].exists()
