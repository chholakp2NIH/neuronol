import os

from dotenv import load_dotenv

from neuronol.io.importer import DataImporter

# Given
load_dotenv()
# fpath_import = os.getenv("EEG_CSV_IMOTIONS_TRIGS")
fpath_import = os.getenv("EEG_CSV_IMOTIONS")
fpath_headcircum = os.getenv("HEADCIRCUM_JSON")
fpath_mne_report = os.getenv("FPATH_MNE_REPORT")

# Read data and display
data_importer = DataImporter(
    fpath_import,
    fpath_headcircum,
    fpath_mne_report=fpath_mne_report,
    verbose=True,
)

# Run full data import
data_importer.run()
# print(data_importer.create_mne_report)

# Test function
# print(data_importer.recording.raw)
