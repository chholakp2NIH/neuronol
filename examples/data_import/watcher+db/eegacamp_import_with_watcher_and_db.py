import os
import sys
import time
from pathlib import Path

DATA_IMPORTS_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_IMPORTS_EXAMPLES_DIR))
from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, import_unimported_eegacamp_data

from neuronol.io.dbmanager import DBManager
from neuronol.io.watcher import DataWatcher

# Given
data_dir = Path(
    os.environ.get(
        "EXAMPLE_DATA_DIR",
        # os.path.expanduser("~/data/bids/imotions-sample/"),
        os.path.expanduser("~/data/bids/eegacamp-test/"),
    )
)
recording_data_dir = data_dir / "sub-xx/ses-studyvisit2/eeg/"
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

# Create data watcher and start scouting
watcher = DataWatcher(data_dir)
watcher.observer.start()
try:
    while True:
        time.sleep(1)
        recording_data_dir = watcher.event_handler.queue.get()
        import_unimported_eegacamp_data(recording_data_dir, db_manager)
        # try:
        #     import_new_eegacamp_data(recording_data_dir, db_manager)
        # except Exception as e:
        #     print(f"\n(xx) Data import failed: {e}")
finally:
    watcher.observer.stop()
    watcher.observer.join()
