import os

from dotenv import load_dotenv

from neuronol.constants import EASYCAP_EEG_CHANNELS
from neuronol.io.importer import DataImporter

# Given
load_dotenv()
# fpath_import = os.getenv("EEG_CSV_IMOTIONS_TRIGS")
# fpath_mne_report = os.getenv("FPATH_MNE_REPORT_TRIGS")
fpath_import = os.getenv("EEG_CSV_IMOTIONS")
fpath_mne_report = os.getenv("FPATH_MNE_REPORT")
fpath_headcircum = os.getenv("HEADCIRCUM_JSON")
fpath_restingstate_stimtimes = os.getenv("FPATH_RESTINGSTATE_STIMTIMES")
fpath_dig = os.getenv("FPATH_DIG")

# Read data and display
fpath_import = fpath_import if fpath_import is not None else ""
fpath_headcircum = fpath_headcircum if fpath_headcircum is not None else ""
data_importer = DataImporter(
    fpath_import,
    fpath_headcircum,
    fpath_mne_report=fpath_mne_report,
    verbose=True,
)

# Run full data import
fpath_dig = fpath_dig if fpath_dig is not None else ""
data_importer.run(
    gnd_channel="GND",
    fpath_bs_dig=fpath_dig,
    renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"],
)

# Test function
# data_importer.create_MNE_montage_from_BS_dig_data(
#     fpath_dig, renamed_channels=EASYCAP_EEG_CHANNELS + ["GND"]
# )
# print(data_importer.recording.dig.ch_names)
