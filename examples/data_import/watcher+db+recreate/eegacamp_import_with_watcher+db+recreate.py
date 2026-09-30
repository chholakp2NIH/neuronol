import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

from neuronol.constants import WATCHER_DATA_COPY_COMPLETED

DATA_IMPORTS_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_IMPORTS_EXAMPLES_DIR))
from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, import_unimported_eegacamp_data

from neuronol.io.dbmanager import DBManager
from neuronol.io.watcher import DataWatcher

# Given
data_dir = Path(
    os.environ.get(
        "EXAMPLE_DATA_DIR",
        os.path.expanduser("~/data/bids/imotions-sample/"),
        # os.path.expanduser("~/data/bids/eegacamp-test/"),
    )
)
fpath_db = data_dir / "derivatives/imaging.db"

# Create arg parser
parser = ArgumentParser()
parser.add_argument(
    "--recreate-all",
    action="store_true",
    help="Delete and rebuild db by re-importing all recordings with a .data_collection_file.",
)
args = parser.parse_args()

# Prepare db
#   Delete db (if exists) when re-importing all
if args.recreate_all:
    print(f"Arg --recreate-all all set to: {args.recreate_all}")
    if fpath_db.exists():
        fpath_db.unlink()
#   Connect to db (if exists); else create db and initialize
if fpath_db.exists():
    db_manager = DBManager(fpath_db)
else:
    db_manager = DBManager(
        fpath_db, initialize_db=True, db_init_script=EEGACAMP_DB_INIT_SCRIPT
    )

# Recreate all
if args.recreate_all:
    # Add recording data dirs with full data to queue
    print("\nImporting the following recording dirs:")
    recording_data_dirs_to_import = []
    for dirpath, dirnames, fnames in sorted(data_dir.walk()):
        if WATCHER_DATA_COPY_COMPLETED in fnames:
            print(dirpath)
            recording_data_dirs_to_import.append(dirpath)
    # Import recordings in queue
    for recording_data_dir in recording_data_dirs_to_import:
        try:
            import_unimported_eegacamp_data(recording_data_dir, db_manager)
        except Exception as e:
            print(f"\n(xx) Data import failed: {e}")
else:  # Normal import mode with watcher
    # Create data watcher and start scouting
    watcher = DataWatcher(data_dir)
    watcher.observer.start()
    try:
        while True:
            time.sleep(1)
            recording_data_dir = watcher.event_handler.queue.get()
            try:
                import_unimported_eegacamp_data(recording_data_dir, db_manager)
            except Exception as e:
                print(f"\n(xx) Data import failed: {e}")
    finally:
        watcher.observer.stop()
        watcher.observer.join()
