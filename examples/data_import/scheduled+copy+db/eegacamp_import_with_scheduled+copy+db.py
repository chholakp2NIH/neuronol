import os
import sys
import time
from pathlib import Path

DATA_IMPORTS_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_IMPORTS_EXAMPLES_DIR))
from _eegacamp import (
    EEGACAMP_DB_INIT_SCRIPT,
    copy_new_recording_data_to_dir,
    import_unimported_eegacamp_data,
)

from neuronol.constants import WATCHER_DATA_COPY_COMPLETED
from neuronol.io.dbmanager import DBManager

# Given
t_sched = 5
# data_dir = Path(
#     os.environ.get(
#         # "EXAMPLE_DATA_DIR", os.path.expanduser("~/data/bids/imotions-sample/")
#         "EXAMPLE_DATA_DIR",
#         os.path.expanduser("~/data/bids/eegacamp-test/"),
#     )
# )
src_dir = Path.home() / "data/bids/eegacamp"
dst_dir = Path.home() / "data/bids/eegacamp-test"
fpath_db = dst_dir / "derivatives/imaging.db"

# Prepare db
if fpath_db.exists():
    db_manager = DBManager(fpath_db)
else:
    db_manager = DBManager(
        fpath_db, initialize_db=True, db_init_script=EEGACAMP_DB_INIT_SCRIPT
    )

while True:
    print()
    # Copy new/unavailable data
    copy_new_recording_data_to_dir(src_dir, dst_dir)
    # Add recording data dirs with full data to queue
    recording_data_dirs_to_import = []
    for dirpath, dirnames, fnames in sorted(dst_dir.walk()):
        if WATCHER_DATA_COPY_COMPLETED in fnames:
            print(dirpath)
            recording_data_dirs_to_import.append(dirpath)
    # Import recordings in queue that are absent in db
    for recording_data_dir in recording_data_dirs_to_import:
        try:
            import_unimported_eegacamp_data(recording_data_dir, db_manager)
        except Exception as e:
            print(f"\n(xx) Data import failed for {recording_data_dir}: {e}")
    time.sleep(t_sched)
