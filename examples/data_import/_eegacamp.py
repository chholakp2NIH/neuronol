from pathlib import Path

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.dbmanager import DBManager
from neuronol.io.importer import DataImporter
from neuronol.utilities import get_sub_and_ses_from_bids_rec_dir

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
    PRIMARY KEY (sub_id, ses_id),
    FOREIGN KEY (sub_id) REFERENCES participants (sub_id),
    FOREIGN KEY (sub_id, ses_id) REFERENCES sessions (sub_id, ses_id)
);
"""  # default db init script


def import_unimported_eegacamp_data(recording_data_dir: Path, db_manager: DBManager):
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
    # existing_sessions_ses_ids = db_manager.read_col_values_from_table(
    #     "sessions", "ses_id"
    # )

    if (sub_id, ses_id) in existing_recs:
        print("\n(!!) Recording already exists in db. Exiting...")
    else:
        print(f"\nImporting recording data for sub-{sub_id} ses-{ses_id}...")

        # Import data
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
        data_importer.run()

        # Add new recording data to db
        #   Table: participants
        if sub_id not in existing_participants_sub_ids:
            db_manager.add_row_to_table(
                "participants", ("sub_id",), [(sub_id,)]
            )  # add participant
        #   Table: sessions
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
                )
            ],
        )  # add recording


def create_bids_file_paths_for_eegacamp_recording(recording_data_dir: Path):
    """
    Creates file paths for raw iMotions' CSV and other related
    files found under the recording data path.
    """
    # subj_id = re.findall(r"^.+/sub-(.+?)/.+$", str(recording_data_dir))[0]
    sub_id, _ = get_sub_and_ses_from_bids_rec_dir(recording_data_dir)
    fpaths = {
        "fpath_import": recording_data_dir / f"sub-{sub_id}_task-all_eeg.csv",
        "fpath_mne_raw": recording_data_dir / f"sub-{sub_id}_task-all_eeg.fif",
        "fpath_mne_report": recording_data_dir
        / f"sub-{sub_id}_task-all_importreport.html",
        "fpath_headcircum": (
            recording_data_dir / f"sub-{sub_id}_desc-manual_headcircumference.json"
        ),
        "fpath_bs_dig": (
            recording_data_dir / f"sub-{sub_id}_task-all_acq-polhemus_headshape.mat"
        ),
        "event_files": recording_data_dir.glob("*_eventsstimtimes.xlsx"),
    }
    return fpaths
