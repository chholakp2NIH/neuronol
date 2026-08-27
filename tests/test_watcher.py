import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from watchdog.events import FileSystemEvent
from watchdog.observers.api import BaseObserver

from neuronol.constants import EASYCAP_EEG_CHANNELS, WATCHER_DATA_COPY_COMPLETED
from neuronol.io.importer import DataImporter
from neuronol.io.watcher import DataEventHandler, DataWatcher
from neuronol.utilities import create_bids_file_paths_for_eegacamp_recording

load_dotenv()


@pytest.fixture
def data_dir():
    value = os.getenv("DATA_DIR")
    assert value is not None
    return value


@pytest.fixture
def fpath_data_completion_flag():
    value = os.getenv("FPATH_DATA_COMPLETION_FLAG")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_report():
    value = os.getenv("FPATH_MNE_REPORT")
    assert value is not None
    return value


@pytest.fixture
def fpath_mne_raw():
    value = os.getenv("FPATH_MNE_RAW")
    assert value is not None
    return value


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


def test_observer_init():
    watcher = DataWatcher(".")
    assert isinstance(watcher.observer, BaseObserver)
    assert isinstance(watcher.event_handler, DataEventHandler)


def test_example_integration(
    data_dir,
    fpath_data_completion_flag,
    fpath_mne_raw,
    fpath_mne_report,
):
    watcher = DataWatcher(data_dir)
    watcher.observer.start()
    # Trigger watcher
    if Path(fpath_mne_raw).exists():
        Path(fpath_mne_raw).unlink()
    if Path(fpath_mne_report).exists():
        Path(fpath_mne_report).unlink()
    if Path(fpath_data_completion_flag).exists():
        Path(fpath_data_completion_flag).unlink()
    Path(fpath_data_completion_flag).touch()
    # Run data import once watcher is triggered
    recording_data_dir = watcher.event_handler.queue.get()
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
    data_importer.run()
    # Stop observer once test completed
    watcher.observer.stop()
    watcher.observer.join()
    # Assertions
    assert Path(fpath_mne_raw).exists()
    assert Path(fpath_mne_report).exists()
