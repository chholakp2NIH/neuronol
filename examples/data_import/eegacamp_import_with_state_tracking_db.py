from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.dbmanager import DBManager
from neuronol.io.importer import DataImporter
from neuronol.utilities import (
    create_bids_file_paths_for_eegacamp_recording,
    get_subj_and_session_from_eegacamp_recording_data_dir,
)

# Given
recording_data_dir = (
    Path.home() / "data/bids/imotions-sample" / "sub-xx/ses-studyvisit2/eeg/"
)
fpath_db = Path.home() / "data/bids/imotions-sample" / "derivatives/imaging.db"
db_init_script = """
PRAGMA foreign_keys = ON;
CREATE TABLE participants (
    sub_id INT PRIMARY KEY
);
CREATE TABLE sessions (
    sub_id INT NOT NULL,
    ses_id TEXT NOT NULL,
    PRIMARY KEY (sub_id, ses_id),
    FOREIGN KEY (sub_id) REFERENCES participants (sub_id)
);
CREATE TABLE recordings (
    sub_id INT NOT NULL,
    ses_id TEXT NOT NULL,
    PRIMARY KEY (sub_id, ses_id),
    FOREIGN KEY (sub_id) REFERENCES participants (sub_id),
    FOREIGN KEY (sub_id, ses_id) REFERENCES sessions (sub_id, ses_id)
);
"""


# Prepare db
# if fpath_db.exists():
#     fpath_db.unlink()
# db_manager = DBManager(fpath_db, initialize_db=True, db_init_script=db_init_script)
db_manager = DBManager(fpath_db)

# Get subj and ses ids from recording data dir
sub_id, ses_id = get_subj_and_session_from_eegacamp_recording_data_dir(
    recording_data_dir
)

# Query db
existing_sub_ids = db_manager.read_col_values_from_table("recordings", "sub_id")
existing_ses_ids = db_manager.read_col_values_from_table("recordings", "ses_id")
existing_recs = list(zip(existing_sub_ids, existing_ses_ids))
# if (sub_id in existing_sub_ids) and (ses_id in existing_ses_ids):
if (sub_id, ses_id) in existing_recs:
    print("\n(!!) Recording already exists in db. Exiting...")
else:
    print(f"\nImporting recording data for sub-{sub_id} ses-{ses_id}...")

    # Read data and display
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

    # Run full data import
    data_importer.run()

    # Add new recording data to db
    if sub_id not in existing_sub_ids:
        db_manager.add_row_to_table(
            "participants", ("sub_id",), [(sub_id,)]
        )  # add participant
    db_manager.add_row_to_table(
        "sessions", ("ses_id", "sub_id"), [(ses_id, sub_id)]
    )  # add session
    db_manager.add_row_to_table(
        "recordings", ("sub_id", "ses_id"), [(sub_id, ses_id)]
    )  # add recording
