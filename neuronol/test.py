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
fpath = (
    Path.home()
    / "data/bids/eegacamp-training/sub-03/ses-studyvisit2/eeg/sub-03_task-all_eeg.csv"
)
# print(fpath)

# Read data and display
data_importer = DataImporter(fpath)
data_importer.read_imotions_csv_as_df()
print(data_importer.df_imotions)

# Get recording date/time
data_importer.get_recording_datetime_from_imotions_df()
print(data_importer.recording_dt)

# Drop header info
data_importer.drop_header_info_from_df()
print(data_importer.df_imotions)
