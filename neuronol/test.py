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
# data_importer.read_imotions_csv_as_df()
# data_importer.read_imotions_data_as_df()
# print(data_importer.recording.df_raw)

# Read preamble and display
# data_importer.read_imotions_csv_preamble()
# print(data_importer.recording.preamble)

# Get recording date/time
# data_importer.get_recording_datetime_from_imotions_preamble()
# print(data_importer.recording_dt)

# Get EEG data column numbers
# data_importer.get_eeg_data_column_numbers()
# print(data_importer.eeg_col_nums)

# Extract just EEG data from iMotions data
# data_importer.extract_eeg_from_imotions_data()
# print(data_importer.df_eeg)

# # Extract head radius from json file
# data_importer.evaluate_head_radius()
# print(
#     f"Head circumference = {data_importer.recording.head_circum:0.3f}m;",
#     f"Head radius = {data_importer.recording.head_radius:0.3f}m",
# )

# Run full iMotions data extractor
data_importer.create_mne_raw_from_imotions_csv()
data_importer.add_interpolated_ecg_to_eeg()
print(data_importer.recording.df_eeg)
# print(data_importer.recording.df_eeg)
