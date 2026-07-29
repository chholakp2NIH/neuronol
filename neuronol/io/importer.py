import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from neuronol.neuronol.constants import (IMOTIONS_ECG_COL,
                                         IMOTIONS_TIMESTAMP_COL)


class Recording:
    fpath_import: Path
    fpath_headcircum: Path
    df_raw: pd.DataFrame
    preamble: str
    recording_dt: dt.datetime
    eeg_col_nums: list[int]
    df_eeg: pd.DataFrame
    head_circum: float
    head_radius: float

    def __init__(self, fpath_import, fpath_headcircum) -> None:
        self.fpath_import = fpath_import
        self.fpath_headcircum = fpath_headcircum


class DataImporter:
    recording: Recording
    verbose: bool

    def __init__(
        self, fpath_import: Path, fpath_headcircum: Path, verbose: bool = True
    ) -> None:
        self.recording = Recording(fpath_import, fpath_headcircum)
        self.verbose = verbose

    def create_mne_raw_from_imotions_csv(self):
        """
        Main script to generate MNE Raw object from imported iMotions CSV.
        """
        # Read head circumference and evaluate head radius
        self.evaluate_head_radius()
        message = (
            f"Head circumference: {self.recording.head_circum * 100:0.1f}cm; "
            + f"Head radius: {self.recording.head_radius:0.3f}m."
        )
        self._log_message(message)

        # Import raw data in CSV file and convert to a formatted dataframe
        self.read_imotions_data_as_df()
        message = f"Read all raw iMotions' data from CSV file"
        self._log_message(message)

        # Read iMotion's CSV file preamble
        self.read_imotions_csv_preamble()
        message = f"Read iMotion's preamble from CSV file"
        self._log_message(message)

        # Get recording datetime from the imported CSV's preamble
        self.get_recording_datetime_from_imotions_preamble()
        message = f"Recording time: {self.recording.recording_dt}"
        self._log_message(message)

        # Extract EEG data from imported iMotions' raw data
        self.extract_eeg_from_imotions_data()
        message = f"EEG data extracted from raw iMotions' data"
        self._log_message(message)

    def read_imotions_csv_full(self) -> None:
        """
        Read the full input iMotion CSV and separate the data and preamble.
        """
        self.read_imotions_data_as_df()
        self.read_imotions_csv_preamble()
        return None

    def read_imotions_data_as_df(self) -> None:
        """
        Read input iMotions CSV data as a DataFrame.
        """
        self.recording.df_raw = pd.read_csv(
            self.recording.fpath_import, comment="#", low_memory=False
        )
        return None

    def read_imotions_csv_preamble(self) -> None:
        """
        Read preamble info from input iMotions CSV data.
        """
        with open(self.recording.fpath_import) as f:
            self.recording.preamble = "".join(
                [line for line in f.readlines() if line.startswith("#")]
            )
        return None

    def evaluate_head_radius(self):
        """
        Reads head circumference (in meters) from file and calculates head radius.
        """
        with open(self.recording.fpath_headcircum, "r") as f:
            content = json.load(f)
        self.recording.head_circum = float(content["Value"])
        self.recording.head_radius = self.recording.head_circum / (
            2 * np.pi
        )  # head circumference / 2π
        return None

    def get_recording_datetime_from_imotions_preamble(self):
        """
        Find the date/time of the iMotions recording from the CSV preamble.
        """
        date_str = time_str = None
        for line in self.recording.preamble.splitlines():
            if line.startswith("#Recording time"):
                m = re.search(r"^.+,Date: (\d{4}-\d{2}-\d{2}),.+$", line)
                if m:
                    date_str = m.group(1)
                m = re.search(r"^.+,Time: (\d{2}:\d{2}:\d{2}\.\d{3}) .+,.+$", line)
                if m:
                    time_str = m.group(1)
                break
        if (date_str is not None) and (time_str is not None):
            self.recording.recording_dt = dt.datetime.fromisoformat(
                f"{date_str} {time_str}"
            )
        else:
            raise ValueError(
                "(xx) Could not find recording date/time from iMotions preamble"
            )
        return None

    def get_eeg_data_column_numbers(self):
        """
        Use the iMotions CSV preamble to find the column numbers that contain
        EEG data.
        """
        for line in self.recording.preamble.splitlines():
            if line.startswith("#Group"):
                self.recording.eeg_col_nums = [
                    i for i, w in enumerate(line.split(sep=",")) if w == "EEG"
                ]
                break
        return None

    def extract_eeg_from_imotions_data(self):
        """
        Extract EEG data from iMotions data. Returns EEG data in Volts.
        """
        self.get_eeg_data_column_numbers()  # Get column numbers corres to EEG data
        df_eeg = self.recording.df_raw.iloc[:, [1] + self.recording.eeg_col_nums].copy()
        df_eeg.dropna(inplace=True)  # drop rows without EEG data
        df_eeg = df_eeg.astype("float")
        df_eeg.iloc[:, 1:] /= 1e6  # convert EEG signal unit from μV to V
        df_eeg.reset_index(drop=True, inplace=True)
        self.recording.df_eeg = df_eeg
        return None

    def add_interpolated_ecg_to_eeg(
        self,
        col_ecg: str = IMOTIONS_ECG_COL,
        col_timestamps: str = IMOTIONS_TIMESTAMP_COL,
        interp="linear",
    ):
        """
        Extracts ECG data from iMotions' raw data, then interpolates it
        using `interp` interpolation to match the timestamps for EEG data
        before adding ECG data to the EEG dataframe.

        Args:
            interp (str): Interpolation method. Can either be 'linear'
                          (default) or 'B-spline'.
        """
        # Extract ECG data
        df_ecg = self.recording.df_raw[[col_timestamps, col_ecg]].copy()
        df_ecg.dropna(inplace=True)  # drop rows without ECG data
        df_ecg = df_ecg.astype("float")
        df_ecg.iloc[:, 1:] /= 1e3  # convert ECG signal unit from mV to V
        df_ecg.reset_index(drop=True, inplace=True)

        # Interpolate ECG data to match EEG timestamps and add to EEG df
        if interp == "linear":
            self.recording.df_eeg[col_ecg] = np.interp(
                self.recording.df_eeg[col_timestamps],
                df_ecg[col_timestamps],
                df_ecg[col_ecg],
            )
        elif interp == "B-spline":
            bspl = make_interp_spline(
                df_ecg[col_timestamps], df_ecg[col_ecg], k=3
            )  # B-spline cubic
            self.recording.df_eeg[col_ecg] = bspl(self.recording.df_eeg["Timestamp"])
        return None

    def _log_message(self, message: str):
        """
        Print given message in the default printing style.
        """
        if self.verbose:
            print("\n!!", message, "\n")
        else:
            pass
