import os
from pathlib import Path

from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, import_unimported_eegacamp_data

from neuronol.io.dbmanager import DBManager

# Given
data_dir = Path(
    os.environ.get(
        "EXAMPLE_DATA_DIR", os.path.expanduser("~/data/bids/imotions-sample/")
    )
)
recording_data_dir = data_dir / "sub-003/ses-studyvisit2/eeg/"
fpath_db = data_dir / "derivatives/imaging.db"

# Prepare db
if fpath_db.exists():
    fpath_db.unlink()
if fpath_db.exists():
    db_manager = DBManager(fpath_db)
else:
    db_manager = DBManager(
        fpath_db, initialize_db=True, db_init_script=EEGACAMP_DB_INIT_SCRIPT
    )

# Import data if not exist already
import_unimported_eegacamp_data(recording_data_dir, db_manager)
