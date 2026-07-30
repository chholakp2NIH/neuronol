import sys
from pathlib import Path

# Add the outer parent directory to the path list
# This corresponds to: /Users/chholakp2/analysis/neuronol
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from neuronol.io.importer import DataImporter

# Given
# fpath = Path.home() / "data/bids/eegacamp/sub-00/ses-studyvisit2/eeg/sub-00_raw.csv"
fpath_import = (
    Path.home()
    / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg/sub-xx_task-all_eeg.csv"
)
fpath_headcircum = (
    Path.home()
    / "data/bids/imotions-sample/sub-xx/ses-studyvisit2/eeg/sub-xx_desc-manual_headcircumference.json"
)
# print(fpath)

# Read data and display
data_importer = DataImporter(
    fpath_import=fpath_import, fpath_headcircum=fpath_headcircum, verbose=True
)

# Run full iMotions data extractor
data_importer.create_mne_raw_from_imotions_csv()

# Test function
data_importer.extract_event_times_from_imotions_data("Blink", 1)
# print(data_importer.recording.df_eeg)
