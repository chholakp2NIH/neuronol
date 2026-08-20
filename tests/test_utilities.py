from pathlib import Path

from neuronol.utilities import create_bids_file_paths_for_eegacamp_recording


def test_create_bids_file_paths_for_eegacamp_recording():
    recording_data_dir = (
        Path.home() / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg"
    )
    fpaths = create_bids_file_paths_for_eegacamp_recording(recording_data_dir)
    assert fpaths["fpath_import"] == (recording_data_dir / "sub-xx_task-all_eeg.csv")
    assert fpaths["fpath_mne_raw"] == (recording_data_dir / "sub-xx_task-all_eeg.fif")
    assert fpaths["fpath_mne_report"] == (
        recording_data_dir / "sub-xx_task-all_importreport.html"
    )
    assert fpaths["fpath_headcircum"] == (
        recording_data_dir / "sub-xx_desc-manual_headcircumference.json"
    )
    assert fpaths["fpath_bs_dig"] == (
        recording_data_dir / "sub-xx_task-all_acq-polhemus_headshape.mat"
    )
    assert list(fpaths["event_files"]) == [
        recording_data_dir / "sub-xx_task-restingstate_eventsstimtimes.xlsx"
    ]
