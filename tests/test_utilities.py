from pathlib import Path

from neuronol.utilities import get_sub_and_ses_from_bids_rec_dir


def test_get_subj_and_session_from_eegacamp_recording_data_dir():
    recording_data_dir = (
        Path.home() / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg"
    )
    sub_id, ses_id = get_sub_and_ses_from_bids_rec_dir(recording_data_dir)
    assert sub_id == "xx"
    assert ses_id == "studyvisit2"
