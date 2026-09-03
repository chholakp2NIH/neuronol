import re
from pathlib import Path


def get_sub_and_ses_from_bids_rec_dir(recording_data_dir: Path):
    """
    Finds subject and session ids from recording data directory following BIDS format.
    """
    sub_id = re.findall(r"^.+/sub-(.+?)/.+$", str(recording_data_dir))[0]
    ses_id = re.findall(r"^.+/ses-(.+?)/.+$", str(recording_data_dir))[0]
    return sub_id, ses_id
