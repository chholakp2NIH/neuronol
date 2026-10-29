from pathlib import Path

from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, add_recording_to_db

from neuronol.io.dbmanager import DBManager

# Given
recording_data_dir = Path.home() / "data/bids/eegacamp" / "sub-002/ses-studyvisit2/eeg/"
# fpath_db = Path.home() / "data/bids/imotions-sample" / "derivatives/imaging.db"
fpath_db = Path.home() / "data/bids/eegacamp" / "derivatives/imaging.db"

# Prepare db
if fpath_db.exists():
    db_manager = DBManager(fpath_db)
else:
    db_manager = DBManager(
        fpath_db, initialize_db=True, db_init_script=EEGACAMP_DB_INIT_SCRIPT
    )

# Add recording to db
add_recording_to_db(
    recording_data_dir,
    db_manager,
    update_db=True,
    is_eeg_imported=True,
    is_ecg_imported=True,
    is_blinks_imported=False,
    is_events_imported=True,
    is_events_trig_imported=False,
    is_events_file_imported=True,
    is_digitization_imported=True,
    is_headcircum_imported=True,
)
