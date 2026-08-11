import os

from dotenv import load_dotenv

from neuronol.io.importer import DataImporter

# Given
load_dotenv()
# fpath_import = os.getenv("EEG_CSV_IMOTIONS_TRIGS")
# fpath_mne_report = os.getenv("FPATH_MNE_REPORT_TRIGS")
fpath_import = os.getenv("EEG_CSV_IMOTIONS")
fpath_mne_report = os.getenv("FPATH_MNE_REPORT")
fpath_headcircum = os.getenv("HEADCIRCUM_JSON")
fpath_restingstate_stimtimes = os.getenv("FPATH_RESTINGSTATE_STIMTIMES")

# Read data and display
data_importer = DataImporter(
    fpath_import,
    fpath_headcircum,
    fpath_mne_report=fpath_mne_report,
    verbose=True,
)

# Run full data import
data_importer.run(event_files=[fpath_restingstate_stimtimes])

# Test function
# data_importer.add_event_markers_from_event_files_to_mne_raw(
#     fpath_restingstate_stimtimes
# )
# print(data_importer.recording.event_onsets)
