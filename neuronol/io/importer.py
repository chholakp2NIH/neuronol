import datetime as dt
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from neuronol.constants import (
    EASYCAP_EEG_CHANNELS,
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
    IMOTIONS_MARKERS_COL,
    IMOTIONS_TIMESTAMP_COL,
    SFREQ,
    T_WIN_NO_DUPL_MARKERS,
)


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
    ecg_data_imported: bool | None = None
    blink_data_imported: bool | None = None
    event_markers_imported: bool | None = None
    blink_times: np.ndarray | None = None
    event_markers: pd.DataFrame | None = None
    raw: mne.io.RawArray

    def __init__(self, fpath_import, fpath_headcircum) -> None:
        self.fpath_import = fpath_import
        self.fpath_headcircum = fpath_headcircum


class DataImporter:
    recording: Recording
    verbose: bool
    report: mne.Report
    create_mne_report: bool | None = None

    def __init__(
        self,
        fpath_import: Path,
        fpath_headcircum: Path,
        fpath_mne_report: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.recording = Recording(fpath_import, fpath_headcircum)
        self.verbose = verbose
        if fpath_mne_report and fpath_mne_report is not None:
            self.fpath_mne_report = fpath_mne_report
            self.create_mne_report = True
        else:
            self.create_mne_report = False

    def run(self):
        """
        Run data import.
        """
        # Initialize MNE report (if needed)
        if self.create_mne_report:
            self.report = mne.Report(
                title=self.recording.fpath_import, verbose=self.verbose
            )

        # Create MNE Raw from iMotions' exported CSV
        self.create_mne_raw_from_imotions_csv()

        # Save MNE report (if created)
        if self.create_mne_report:
            self.report.save(self.fpath_mne_report, overwrite=True)

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
        if self.create_mne_report:
            self.report.add_html(title="Head radius import", html=message)

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

        # Extract ECG signal (Channel: "ECG LL-RA CAL"; in millivolts)
        if IMOTIONS_ECG_COL in self.recording.df_raw.columns:
            try:
                self.add_interpolated_ecg_to_eeg()
                self.recording.ecg_data_imported = True
                message = "Successfully imported ECG data."
            except Exception as e:
                self.recording.ecg_data_imported = False
                message = e
        else:
            self.recording.ecg_data_imported = False
            message = f"ECG column ('{IMOTIONS_ECG_COL}' not found in raw data from iMotions.)"
        self._log_message(message)
        if self.create_mne_report:
            self.report.add_html(title="ECG data import", html=message)

        # Extract blink times
        if IMOTIONS_BLINK_COL in self.recording.df_raw.columns:
            try:
                self.extract_event_times_from_imotions_data(
                    IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
                )
                self.recording.blink_data_imported = True
                message = "Successfully imported blink data."
            except Exception as e:
                self.recording.blink_data_imported = False
                message = e
        else:
            self.recording.blink_data_imported = False
            message = f"Blink column ('{IMOTIONS_BLINK_COL}' not found in raw data from iMotions.)"
        self._log_message(message)
        if self.create_mne_report:
            self.report.add_html(title="Blink data import", html=message)

        # Extract event markers
        if IMOTIONS_MARKERS_COL in self.recording.df_raw.columns:
            try:
                self.read_event_markers_from_imotions_data()
                self.recording.event_markers_imported = True
                message = "Successfully imported event markers."
            except Exception as e:
                self.recording.event_markers_imported = False
                message = e
        else:
            self.recording.event_markers_imported = False
            message = f"Event marker column ('{IMOTIONS_MARKERS_COL}' not found in raw data from iMotions.)"
        self._log_message(message)
        if self.create_mne_report:
            self.report.add_html(title="Event markers import", html=message)

        # Convert EEG/ECG dataframe to MNE Raw object
        self.convert_electrophys_data_to_mne_raw_object()
        message = f"EEG/ECG data converted to MNE Raw object."
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
                "(xx) Could not find recording date/time from iMotions preamble."
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

    def extract_eeg_from_imotions_data(
        self,
        eeg_channels: list[str] = EASYCAP_EEG_CHANNELS,
    ):
        """
        Extract EEG data from iMotions data. Returns EEG data in Volts.
        """
        self.get_eeg_data_column_numbers()  # Get column numbers corres to EEG data
        df_eeg = self.recording.df_raw.iloc[:, self.recording.eeg_col_nums].copy()
        df_eeg.index = self.recording.df_raw[IMOTIONS_TIMESTAMP_COL].astype(float)
        df_eeg.dropna(inplace=True)  # drop rows without EEG data
        df_eeg = df_eeg.astype(float)
        df_eeg /= 1e6  # convert EEG signal unit from μV to V

        # Ensure channels correctly identified and named
        msgs = []
        if len(df_eeg.columns) > len(
            eeg_channels
        ):  # iMotions labels AUX channels as EEG in some versions
            df_eeg = df_eeg.iloc[:, : len(eeg_channels)]
            msgs.append("Dropped extra misidentified EEG channels.")
        else:
            msgs.append("Correct number of EEG channels found in iMotions' CSV.")
        if df_eeg.columns.tolist() != eeg_channels:
            df_eeg.columns = eeg_channels
            msgs.append("Renamed electrophys channel names.")
        else:
            msgs.append("Channels correctly named in iMotions' CSV.")
        message = " ".join(msgs)
        self._log_message(message)
        if self.create_mne_report:
            self.report.add_html(title="EEG data import", html=message)

        self.recording.df_eeg = df_eeg

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
        ecg = self.recording.df_raw[col_ecg].copy()
        ecg.index = self.recording.df_raw[col_timestamps].astype(float)
        ecg.dropna(inplace=True)  # drop rows without ECG data
        ecg = ecg.astype("float")
        ecg /= 1e3  # convert ECG signal unit from mV to V
        # Interpolate ECG data to match EEG timestamps and add to EEG df
        if interp == "linear":
            self.recording.df_eeg[col_ecg] = np.interp(
                self.recording.df_eeg.index,
                ecg.index,
                ecg.values,
            )
        elif interp == "B-spline":
            bspl = make_interp_spline(ecg.index, ecg.values, k=3)  # B-spline cubic
            self.recording.df_eeg[col_ecg] = bspl(self.recording.df_eeg.index)
        return None

    def extract_event_times_from_imotions_data(self, col_event, positive_val_event):
        """
        Find times in seconds corresponding to event; reports times
        at which event column has a value that corresponds to event happening.
        """
        df_event = self.recording.df_raw[[IMOTIONS_TIMESTAMP_COL, col_event]].copy()
        df_event = df_event[df_event[col_event] == positive_val_event]
        self.recording.blink_times = (
            df_event[IMOTIONS_TIMESTAMP_COL]
            - self.recording.df_raw[IMOTIONS_TIMESTAMP_COL][0]
        ) / 1000  # in seconds

    def read_event_markers_from_imotions_data(
        self,
        t_win_no_dupl_markers=T_WIN_NO_DUPL_MARKERS,
        marker_col=IMOTIONS_MARKERS_COL,
    ):
        """
        Reads LSL markers from raw dataframe based on iMotions' exported CSV data.
        Drops trigs that occur sooner than `t_win_no_dupl_markers` time (in s) after
        an existing trig of the same name.
        """
        # Create a df for markers
        df_markers = self.recording.df_raw[[IMOTIONS_TIMESTAMP_COL, marker_col]].copy()
        df_markers.dropna(inplace=True)
        df_markers[IMOTIONS_TIMESTAMP_COL] = df_markers[IMOTIONS_TIMESTAMP_COL].astype(
            float
        )
        # Drop rows with duplicate trigger entries
        df_markers["Timestamp_secs"] = (
            df_markers[IMOTIONS_TIMESTAMP_COL] / 1000
        )  # in seconds
        df_markers["TimestampDiff_secs"] = df_markers[
            "Timestamp_secs"
        ].diff()  # differences between subsequent time stamps (in secs)
        df_markers["MarkerSameAsLast"] = df_markers[marker_col] == df_markers[
            marker_col
        ].shift(periods=1)
        df_markers = df_markers[
            ~(
                (df_markers["TimestampDiff_secs"] < t_win_no_dupl_markers)
                & (df_markers["MarkerSameAsLast"])
            )
        ]
        t_0 = (
            self.recording.df_raw[IMOTIONS_TIMESTAMP_COL][0] / 1000
        )  # timestamp corres to start of imotions recording (in secs)
        df_markers["Times"] = df_markers["Timestamp_secs"] - t_0
        self.recording.event_markers = df_markers[["Times", IMOTIONS_MARKERS_COL]]

    def convert_electrophys_data_to_mne_raw_object(self, sfreq: float = SFREQ):
        """
        Converts recorded EEG/ECG data to MNE Raw object.
        """
        # Convert df to np array
        data = self.recording.df_eeg.values.T

        # Create MNE Info object
        ch_names = self.recording.df_eeg.columns.tolist()
        if self.recording.ecg_data_imported:
            ch_types = ["eeg"] * (len(ch_names) - 1) + ["ecg"]
        else:
            ch_types = ["eeg"] * len(ch_names)
        info = mne.create_info(ch_names, sfreq, ch_types=ch_types, verbose=self.verbose)

        # Create MNE Raw obj
        self.recording.raw = mne.io.RawArray(data, info)

    def read_event_markers_from_event_files(self): ...

    # def read_event_times_from_task(self): ...

    def _log_message(self, message: str | Exception):
        """
        Print given message in the default printing style.
        """
        if self.verbose:
            print("\n!!", message, "\n")
        else:
            pass
