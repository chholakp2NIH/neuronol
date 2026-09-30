import re
from datetime import datetime
from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.dbmanager import DBManager
from neuronol.io.importer import DataImporter
from neuronol.utilities import (
    create_bids_fpaths_from_recording_dir,
    get_sub_and_ses_from_bids_rec_dir,
)

EEGACAMP_DB_INIT_SCRIPT = """
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
    is_eeg_imported BOOLEAN,
    is_ecg_imported BOOLEAN,
    is_blinks_imported BOOLEAN,
    is_events_imported BOOLEAN,
    is_digitization_imported BOOLEAN,
    is_headcircum_imported BOOLEAN,
    qc_passed BOOLEAN,
    imported_on TEXT,
    PRIMARY KEY (sub_id, ses_id),
    FOREIGN KEY (sub_id) REFERENCES participants (sub_id),
    FOREIGN KEY (sub_id, ses_id) REFERENCES sessions (sub_id, ses_id)
);
"""  # default db init script


def copy_new_recording_data_to_dir(src_dir: Path, dst_dir: Path):
    """
    Copies all recording data from source data directory to destination
    data directory unless they already exist at destination.
    """
    # Create list of recordings at source
    recs_src = []
    for root, dirs, _ in src_dir.walk():
        if "derivatives" in dirs:
            dirs.remove("derivatives")
        if "ses-practice" in dirs:
            dirs.remove("ses-practice")
        if re.match(r"^.+/EA-\d{3}/ses-.+/eeg$", str(root)):
            sub_id = int(re.findall(r"^.+/EA-(\d{3})/ses-.+/eeg", str(root))[0])
            ses_id = re.findall(r"^.+/EA-\d{3}/ses-(.+)/eeg", str(root))[0]
            recs_src.append((sub_id, ses_id))
    # print(sorted(recs_src))

    # Create recording paths at the destination
    for sub_id, ses_id in sorted(recs_src):
        path_rec_src = src_dir / f"EA-{sub_id:03d}/ses-{ses_id}/eeg"
        path_rec_dst = dst_dir / f"sub-{sub_id:03d}/ses-{ses_id}/eeg"
        path_rec_dst.mkdir(exist_ok=True, parents=True)
        #   Filter: copy only recordings from source which are unavailable at destination
        if (
            len(
                [
                    item
                    for item in path_rec_dst.iterdir()
                    if not item.name.startswith(".")
                ]
            )
            > 0
        ):
            print(f"Recording already exists: {path_rec_dst}. Skipping...")
        else:
            print(f"Copying recording: {path_rec_dst}")
            path_rec_dst.mkdir(parents=True, exist_ok=True)
            for item in path_rec_src.iterdir():
                item.copy(path_rec_dst / item.name.replace("EA-", "sub-", 1))
            #   Create an empty file marking completion of data copying
            path_data_completion_flag = path_rec_dst / ".data_collection_completed"
            path_data_completion_flag.touch()


def import_unimported_eegacamp_data(
    recording_data_dir: Path, db_manager: DBManager, update_db: bool = False
):
    """
    Import recorded data only if it hasn't been imported previously,
    as tracked by the db.
    """
    # Get subj and ses ids from recording data dir
    sub_id_str, ses_id = get_sub_and_ses_from_bids_rec_dir(recording_data_dir)
    sub_id = int(sub_id_str)

    # Query db
    existing_recs_sub_ids = db_manager.read_col_values_from_table(
        "recordings", "sub_id"
    )
    existing_recs_ses_ids = db_manager.read_col_values_from_table(
        "recordings", "ses_id"
    )
    existing_recs = list(zip(existing_recs_sub_ids, existing_recs_ses_ids))
    existing_participants_sub_ids = db_manager.read_col_values_from_table(
        "participants", "sub_id"
    )

    if (sub_id, ses_id) in existing_recs and not update_db:
        print("\n(!!) Recording already exists in db. Exiting...")
        return
    elif (sub_id, ses_id) in existing_recs and update_db:
        print(f"\nUpdating recording data for sub-{sub_id} ses-{ses_id}...")
    else:
        print(f"\nImporting recording data for sub-{sub_id} ses-{ses_id}...")

    # Import data
    bids_fpaths = create_bids_fpaths_from_recording_dir(recording_data_dir)
    data_importer = DataImporter(
        bids_fpaths["imotions-csv"],
        fpath_mne_raw=bids_fpaths["mne-raw"],
        fpath_mne_report=bids_fpaths["mne-report"],
        fpath_headcircum=bids_fpaths["headcircum"],
        fpath_bs_dig=bids_fpaths["headshape-dig"],
        event_files=bids_fpaths["events-stim-times-files"],
        gnd_channel="GND",
        renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
        verbose=True,
    )
    data_importer.run()

    # Add/update new recording data to db
    #   Table: participants
    if sub_id not in existing_participants_sub_ids:
        db_manager.add_row_to_table(
            "participants", ("sub_id",), [(sub_id,)]
        )  # add participant

    #   Table: sessions
    if (sub_id, ses_id) not in existing_recs:
        db_manager.add_row_to_table(
            "sessions",
            ("sub_id", "ses_id"),
            [(sub_id, ses_id)],
        )  # add session

    #   Table: recordings
    is_eeg_imported = True if data_importer.recording.raw is not None else False
    is_ecg_imported = data_importer.recording.ecg_data_imported
    is_blinks_imported = data_importer.recording.blink_data_imported
    is_events_imported = data_importer.recording.event_markers_imported
    is_digitization_imported = (
        True if data_importer.recording.dig is not None else False
    )
    is_headcircum_imported = (
        True if data_importer.recording.head_circum is not None else False
    )
    if (sub_id, ses_id) not in existing_recs:
        db_manager.add_row_to_table(
            "recordings",
            (
                "sub_id",
                "ses_id",
                "is_eeg_imported",
                "is_ecg_imported",
                "is_blinks_imported",
                "is_events_imported",
                "is_digitization_imported",
                "is_headcircum_imported",
                "imported_on",
            ),
            [
                (
                    sub_id,
                    ses_id,
                    is_eeg_imported,
                    is_ecg_imported,
                    is_blinks_imported,
                    is_events_imported,
                    is_digitization_imported,
                    is_headcircum_imported,
                    datetime.now().isoformat(),
                )
            ],
        )  # add recording
    elif (sub_id, ses_id) in existing_recs and update_db:
        db_manager.update_row_in_table(
            "recordings",
            (
                "is_eeg_imported",
                "is_ecg_imported",
                "is_blinks_imported",
                "is_events_imported",
                "is_digitization_imported",
                "is_headcircum_imported",
                "imported_on",
            ),
            [
                (
                    is_eeg_imported,
                    is_ecg_imported,
                    is_blinks_imported,
                    is_events_imported,
                    is_digitization_imported,
                    is_headcircum_imported,
                    datetime.now().isoformat(),
                )
            ],
            (
                "sub_id",
                "ses_id",
            ),
            [(sub_id, ses_id)],
        )  # update recording
    else:
        raise ValueError("Unexpected if/else condition seen")
