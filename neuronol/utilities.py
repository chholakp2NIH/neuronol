import re
from pathlib import Path


def get_sub_and_ses_from_bids_rec_dir(recording_data_dir: Path):
    """
    Finds subject and session ids from recording data directory following BIDS format.
    """
    sub_id = re.findall(r"^.+/sub-(.+?)/.+$", str(recording_data_dir))[0]
    ses_id = re.findall(r"^.+/ses-(.+?)/.+$", str(recording_data_dir))[0]
    return sub_id, ses_id


def create_bids_fpaths_from_recording_dir(
    recording_data_dir: Path,
    fname_suffix_templates={
        "imotions-csv": "_task-all_eeg.csv",
        "mne-raw": "_task-all_eeg.fif",
        "mne-report": "_task-all_importreport.html",
        "headcircum": "_desc-manual_headcircumference.json",
        "headshape-dig": "_task-all_acq-polhemus_headshape.mat",
    },
    fname_glob_templates={"events-stim-times-files": "*_eventsstimtimes.xlsx"},
):
    """
    Creates file paths for raw iMotions' CSV and other related
    files found under the recording data path.
    """
    sub_id, _ = get_sub_and_ses_from_bids_rec_dir(recording_data_dir)
    fpaths = {}
    for ftype in fname_suffix_templates:
        fpaths[ftype] = (
            recording_data_dir / f"sub-{sub_id}{fname_suffix_templates[ftype]}"
        )
    for ftype in fname_glob_templates:
        fpaths[ftype] = recording_data_dir.glob(fname_glob_templates[ftype])
    return fpaths
