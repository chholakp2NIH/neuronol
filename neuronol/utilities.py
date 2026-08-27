import re
from pathlib import Path


def create_bids_file_paths_for_eegacamp_recording(recording_data_dir: Path):
    """
    Creates file paths for raw iMotions' CSV and other related
    files found under the recording data path.
    """
    subj_id = re.findall(r"^.+/sub-(..)/.+$", str(recording_data_dir))[0]
    fpaths = {
        "fpath_import": recording_data_dir / f"sub-{subj_id}_task-all_eeg.csv",
        "fpath_mne_raw": recording_data_dir / f"sub-{subj_id}_task-all_eeg.fif",
        "fpath_mne_report": recording_data_dir
        / f"sub-{subj_id}_task-all_importreport.html",
        "fpath_headcircum": (
            recording_data_dir / f"sub-{subj_id}_desc-manual_headcircumference.json"
        ),
        "fpath_bs_dig": (
            recording_data_dir / f"sub-{subj_id}_task-all_acq-polhemus_headshape.mat"
        ),
        "event_files": recording_data_dir.glob("*_eventsstimtimes.xlsx"),
    }
    return fpaths


def get_subj_and_session_from_eegacamp_recording_data_dir(recording_data_dir: Path):
    """
    Finds subject and session ids from recording data directory following BIDS format.
    """
    sub_id = re.findall(r"^.+/sub-(..)/.+$", str(recording_data_dir))[0]
    ses_id = re.findall(r"^.+/ses-(.+)/.+$", str(recording_data_dir))[0]
    return sub_id, ses_id
