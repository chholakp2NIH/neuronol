import os
from argparse import ArgumentParser
from pathlib import Path

from _eegacamp import EEGACAMP_DB_INIT_SCRIPT, import_unimported_eegacamp_data

from neuronol.constants import WATCHER_DATA_COPY_COMPLETED
from neuronol.io.dbmanager import DBManager

# Given
data_dir = Path(
    os.environ.get(
        # "EXAMPLE_DATA_DIR", os.path.expanduser("~/data/bids/imotions-sample/")
        "EXAMPLE_DATA_DIR",
        os.path.expanduser("~/data/bids/eegacamp-test/"),
    )
)
fpath_db = data_dir / "derivatives/imaging.db"

# Create arg parser
parser = ArgumentParser()
parser.add_argument(
    "--reimport-all",
    action="store_true",
    help="Re-import all recordings with a .data_collection_file.",
)
args = parser.parse_args()

# Prepare db
if fpath_db.exists():
    db_manager = DBManager(fpath_db)
else:
    db_manager = DBManager(
        fpath_db, initialize_db=True, db_init_script=EEGACAMP_DB_INIT_SCRIPT
    )

# Add recording data dirs with full data to queue
recording_data_dirs_to_import = []
for dirpath, dirnames, fnames in sorted(data_dir.walk()):
    if WATCHER_DATA_COPY_COMPLETED in fnames:
        print(dirpath)
        recording_data_dirs_to_import.append(dirpath)

# Import data in queue
for recording_data_dir in recording_data_dirs_to_import:
    try:
        if args.reimport_all:
            import_unimported_eegacamp_data(
                recording_data_dir, db_manager, update_db=True
            )
        else:
            import_unimported_eegacamp_data(recording_data_dir, db_manager)
    except Exception as e:
        print(f"\n(xx) Data import failed for {recording_data_dir}: {e}")
