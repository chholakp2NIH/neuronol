from pathlib import Path, PosixPath

from neuronol.utilities import (
    create_bids_fpaths_from_recording_dir,
    get_sub_and_ses_from_bids_rec_dir,
)


def test_get_subj_and_session_from_eegacamp_recording_data_dir():
    recording_data_dir = (
        Path.home() / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg"
    )
    sub_id, ses_id = get_sub_and_ses_from_bids_rec_dir(recording_data_dir)
    assert sub_id == "xx"
    assert ses_id == "studyvisit2"


def test_create_bids_fpaths_from_recording_dir():
    recording_data_dir = (
        Path.home() / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg"
    )
    fpaths = create_bids_fpaths_from_recording_dir(recording_data_dir)
    print(fpaths)
    assert fpaths["imotions-csv"] == recording_data_dir / "sub-xx_task-all_eeg.csv"
    assert fpaths["mne-raw"] == recording_data_dir / "sub-xx_task-all_eeg.fif"
    assert (
        fpaths["mne-report"] == recording_data_dir / "sub-xx_task-all_importreport.html"
    )
    assert (
        fpaths["headcircum"]
        == recording_data_dir / "sub-xx_desc-manual_headcircumference.json"
    )
    assert (
        fpaths["headshape-dig"]
        == recording_data_dir / "sub-xx_task-all_acq-polhemus_headshape.mat"
    )
    events_stim_times_files = list(fpaths["events-stim-times-files"])
    assert all([isinstance(w, PosixPath) for w in events_stim_times_files])
    assert events_stim_times_files == [
        recording_data_dir / "sub-xx_task-restingstate_eventsstimtimes.xlsx"
    ]
