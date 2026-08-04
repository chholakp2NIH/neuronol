import os
from pathlib import Path

from dotenv import load_dotenv

from neuronol.io.importer import DataImporter

# Given
load_dotenv()
fpath_import = os.getenv("EEG_CSV_IMOTIONS_TRIGS")
fpath_headcircum = os.getenv("HEADCIRCUM_JSON")

# Read data and display
data_importer = DataImporter(
    fpath_import=fpath_import, fpath_headcircum=fpath_headcircum, verbose=True
)

# Run full iMotions data extractor
data_importer.create_mne_raw_from_imotions_csv()

# Test function
# data_importer.read_event_markers_from_imotions_data()
df_markers = data_importer.read_event_markers_from_imotions_data()
print(df_markers)
# print(data_importer.recording.df_eeg)
