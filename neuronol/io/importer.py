import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat

from neuronol.constants import (
    BS_LABEL_LPA,
    BS_LABEL_NAS,
    BS_LABEL_RPA,
    BS_SUBVAR_CHANNEL_LOC,
    BS_SUBVAR_CHANNEL_NAME,
    BS_SUBVAR_HEADPOINTS_LABEL,
    BS_SUBVAR_HEADPOINTS_LOC,
    BS_SUBVAR_HEADPOINTS_TYPE,
    BS_VAR_CHANNEL,
    BS_VAR_HEADPOINTS,
    EASYCAP_EEG_CHANNELS,
    IMOTIONS_BLINK_COL,
    IMOTIONS_BLINK_COL_POSITIVE_VALUE,
    IMOTIONS_ECG_COL,
    IMOTIONS_MARKERS_COL,
    IMOTIONS_TIMESTAMP_COL,
    SFREQ,
    T_WIN_NO_DUPL_MARKERS,
)


@dataclass
class Recording:
    """
    A class representing an iMotions' recording and collected data therein.
    """

    fpath_import: Path

    df_raw: pd.DataFrame | None = None
    preamble: str | None = None
    recording_dt: dt.datetime | None = None
    eeg_col_nums: list[int] | None = None
    df_eeg: pd.DataFrame | None = None
    head_circum: float | None = None
    head_radius: float | None = None

    ecg_data_imported: bool | None = None
    blink_data_imported: bool | None = None
    event_markers_imported: bool | None = None

    blink_times: np.ndarray | None = None
    event_markers: pd.DataFrame | None = None

    # Using `field()` ensures new list is created each time a Recording obj is created
    event_onsets: list[float] = field(default_factory=list)
    event_descriptions: list[str] = field(default_factory=list)

    raw: mne.io.RawArray | None = None
    dig: mne.channels.DigMontage | None = None


class DataImporter:
    recording: Recording
    verbose: bool
    report: mne.Report
    create_mne_report: bool | None = None

    def __init__(
        self,
        fpath_import: Path | str,
        fpath_mne_raw: Path | str | None = None,
        fpath_mne_report: Path | str | None = None,
        fpath_headcircum: Path | str | None = None,
        fpath_bs_dig: Path | str | None = None,
        event_files: list[Path | str] | None = None,
        gnd_channel: str | None = None,
        renamed_channels: list[str] | None = None,
        include_3d_plots: bool = False,
        verbose: bool = True,
    ) -> None:
        self.recording = Recording(Path(fpath_import))
        if fpath_mne_raw is not None:
            self.fpath_mne_raw = Path(fpath_mne_raw)
            self.save_mne_raw = True
        else:
            self.fpath_mne_raw = None
            self.save_mne_raw = False
        if fpath_mne_report is not None:
            self.fpath_mne_report = Path(fpath_mne_report)
            self.create_mne_report = True
        else:
            self.fpath_mne_report = None
            self.create_mne_report = False
        self.fpath_headcircum = (
            Path(fpath_headcircum) if fpath_headcircum is not None else None
        )
        self.fpath_bs_dig = Path(fpath_bs_dig) if fpath_bs_dig is not None else None
        self.event_files = (
            [Path(w) for w in event_files] if event_files is not None else None
        )
        self.gnd_channel = gnd_channel
        self.renamed_channels = renamed_channels
        self.include_3d_plots = include_3d_plots
        self.verbose = verbose

    def run(self) -> None:
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

        if isinstance(self.recording.raw, mne.io.RawArray):
            # Add GND channel to MNE Raw (if needed)
            if self.gnd_channel is not None:
                mne.add_reference_channels(
                    self.recording.raw, ref_channels=[self.gnd_channel], copy=False
                )

            # Add digitized head points from BrainStorm (if needed)
            if self.fpath_bs_dig is not None:
                try:
                    self.create_mne_montage_from_brainstorm_dig_data(
                        self.fpath_bs_dig, renamed_channels=self.renamed_channels
                    )
                    self.recording.raw.set_montage(self.recording.dig)
                    message = "Successfully imported digitized data."
                except Exception as e:
                    message = f"Could not import digitized data:\n{e}"
                self._log_message(message)
                if self.create_mne_report:
                    self.report.add_html(title="Digitization data import", html=message)
                    if isinstance(self.recording.dig, mne.channels.DigMontage):
                        if self.fpath_headcircum is not None:
                            self.evaluate_head_radius()
                            # 2D Matplotlib plot
                            try:
                                message = f"Successfully read head radius and circumference from file."
                            except Exception as e:
                                message = f"Head radius and/or circumference could not be read from file:\n{e}"
                            self._log_message(message)
                            self.report.add_html(
                                title="Head radius import", html=message
                            )
                            if self.recording.head_radius is not None:
                                fig = self.recording.dig.plot(
                                    sphere=self.recording.head_radius
                                )  # 2D plot
                                self.report.add_figure(
                                    fig,
                                    "2D Headshape Digitization",
                                    section="EEG Shape and Headshape Digitization",
                                )
                        if self.include_3d_plots:
                            # 3D Matplotlib plot
                            fig = self.recording.dig.plot(kind="3d")  # 3d plot
                            self.report.add_figure(
                                fig,
                                "3D Headshape Digitization",
                                section="EEG Shape and Headshape Digitization",
                            )
                            # 3D PyVista plot
                            fig = mne.viz.plot_alignment(
                                info=self.recording.raw.info,
                                dig=True,
                            )
                            fig.plotter.camera.elevation = 20
                            fig.plotter.camera.azimuth = 45
                            self.report.add_figure(
                                fig,
                                "3D Headshape Digitization (in PyVista)",
                                section="EEG Shape and Headshape Digitization",
                            )

            # Add events to Raw from Excel files
            if not self.recording.event_markers_imported and self.event_files:
                for fpath in self.event_files:
                    self.add_event_markers_from_event_files_to_mne_raw(fpath)
                self.recording.event_markers_imported = True
                message = f"Successfully read event markers from file(s)."
                self._log_message(message)
                if self.create_mne_report:
                    self.report.add_html(
                        title="Event markers: from file(s)",
                        section="Event markers import",
                        html=message,
                    )
                    events, event_id = mne.events_from_annotations(self.recording.raw)
                    self.report.add_events(
                        events,
                        f"Events: after including event markers read from file(s)",
                        section="Events",
                        event_id=event_id,
                        sfreq=self.recording.raw.info["sfreq"],
                    )

            # Save MNE Raw to disk
            if self.save_mne_raw:
                self.recording.raw.save(self.fpath_mne_raw, overwrite=True)

        # Save MNE report (if created)
        if self.create_mne_report:
            self.report.save(self.fpath_mne_report, overwrite=True)

    def create_mne_raw_from_imotions_csv(self):
        """
        Main script to generate MNE Raw object from imported iMotions CSV.
        """

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
                self.recording.blink_times = (
                    self.extract_event_times_from_imotions_data(
                        IMOTIONS_BLINK_COL, IMOTIONS_BLINK_COL_POSITIVE_VALUE
                    )
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
            self.report.add_html(
                title="Event markers: from embedded triggers",
                section="Event markers import",
                html=message,
            )

        # Convert EEG/ECG dataframe to MNE Raw object
        self.convert_electrophys_data_to_mne_raw_object()
        message = f"EEG/ECG data converted to MNE Raw object."
        self._log_message(message)

        # Add blinks and/or event markers to MNE Raw as events/annotations
        if self.recording.blink_data_imported or self.recording.event_markers_imported:
            self.add_read_blinks_and_event_markers_to_mne_raw()
            if self.create_mne_report and isinstance(
                self.recording.raw, mne.io.RawArray
            ):
                events, event_id = mne.events_from_annotations(self.recording.raw)
                self.report.add_events(
                    events,
                    "Events: read from embedded blinks and/or event markers",
                    section="Events",
                    event_id=event_id,
                    sfreq=self.recording.raw.info["sfreq"],
                )

    def read_imotions_csv_full(self) -> None:
        """
        Read the full input iMotion CSV and separate the data and preamble.
        """
        self.read_imotions_data_as_df()
        self.read_imotions_csv_preamble()

    def read_imotions_data_as_df(self) -> None:
        """
        Read input iMotions CSV data as a DataFrame.
        """
        self.recording.df_raw = pd.read_csv(
            self.recording.fpath_import, comment="#", low_memory=False
        )

    def read_imotions_csv_preamble(self) -> None:
        """
        Read preamble info from input iMotions CSV data.
        """
        with open(self.recording.fpath_import) as f:
            self.recording.preamble = "".join(
                [line for line in f.readlines() if line.startswith("#")]
            )

    def get_recording_datetime_from_imotions_preamble(self) -> None:
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

    def get_eeg_data_column_numbers(self) -> None:
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
    ) -> None:
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
        else:
            raise ValueError(
                f"Unknown interpolation method: {interp}. "
                "Use `linear` or `B-spline`."
            )

    def extract_event_times_from_imotions_data(self, col_event, positive_val_event):
        """
        Find times in seconds corresponding to event; reports times
        at which event column has a value that corresponds to event happening.
        """
        df_event = self.recording.df_raw[[IMOTIONS_TIMESTAMP_COL, col_event]].copy()
        df_event = df_event[df_event[col_event] == positive_val_event]
        event_onsets = (
            df_event[IMOTIONS_TIMESTAMP_COL]
            - self.recording.df_raw[IMOTIONS_TIMESTAMP_COL].iloc[0]
        ) / 1000  # in seconds
        # self.recording.blink_times = blink_onsets
        self.recording.event_onsets += [w for w in event_onsets]
        self.recording.event_descriptions += [col_event] * len(event_onsets)
        return event_onsets

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
            self.recording.df_raw[IMOTIONS_TIMESTAMP_COL].iloc[0] / 1000
        )  # timestamp corres to start of imotions recording (in secs)
        df_markers["Times"] = df_markers["Timestamp_secs"] - t_0
        self.recording.event_markers = df_markers[["Times", IMOTIONS_MARKERS_COL]]
        self.recording.event_onsets += df_markers["Times"].tolist()
        self.recording.event_descriptions += df_markers[IMOTIONS_MARKERS_COL].tolist()

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

    def add_read_blinks_and_event_markers_to_mne_raw(self):
        """
        Adds the blinks and/or event markers read from iMotions' CSV
        (if any) as events/annotations to the created MNE Raw object.
        """
        annots = mne.Annotations(
            onset=self.recording.event_onsets,
            duration=0,
            description=self.recording.event_descriptions,
        )
        self.recording.raw.set_annotations(self.recording.raw.annotations + annots)

    def add_event_markers_from_event_files_to_mne_raw(
        self, fpath_event_markers_file: Path
    ):
        """
        Read event markers from Excel, with columns:
            `EventName`: Event labels/descriptions
            `EventTime`: Event timestamps
        Converts read event markers to datetime and subtracts the recording datetime
        to finally produce event times and descriptions similar to event triggers.
        """
        df = pd.read_excel(fpath_event_markers_file)
        event_descriptions = df["EventName"].tolist()
        event_datetimes = [dt.datetime.fromtimestamp(ts) for ts in df["EventTime"]]
        event_onsets = [
            (dt - self.recording.recording_dt).total_seconds() for dt in event_datetimes
        ]
        annots = mne.Annotations(
            onset=event_onsets,
            duration=0,
            description=event_descriptions,
        )
        self.recording.raw.set_annotations(self.recording.raw.annotations + annots)

    # def read_event_times_from_task(self): ...

    def create_mne_montage_from_brainstorm_dig_data(
        self,
        fpath_dig: Path,
        renamed_channels: list[str] | None = None,
    ):
        """
        Reads Brainstorm Digitizer data (mat-file) to generate a MNE montage object.
        """
        # Load digitized data from file
        data = loadmat(str(fpath_dig), simplify_cells=True)
        # Extract channel, headshape, and fiducial locations
        #   Channel locations
        df_channels = pd.DataFrame(data[BS_VAR_CHANNEL])
        if renamed_channels is not None:
            df_channels[BS_SUBVAR_CHANNEL_NAME] = renamed_channels
        ch_pos = {
            u: v
            for (u, v) in df_channels[
                [BS_SUBVAR_CHANNEL_NAME, BS_SUBVAR_CHANNEL_LOC]
            ].to_numpy()
        }
        #   Head shape and fiducial points
        df_headpoints = pd.concat(
            [
                pd.DataFrame(
                    data[BS_VAR_HEADPOINTS][BS_SUBVAR_HEADPOINTS_LOC].T,
                    columns=["X", "Y", "Z"],
                ),
                pd.DataFrame(
                    data[BS_VAR_HEADPOINTS][BS_SUBVAR_HEADPOINTS_LABEL],
                    columns=["Label"],
                ),
                pd.DataFrame(
                    data[BS_VAR_HEADPOINTS][BS_SUBVAR_HEADPOINTS_TYPE], columns=["Type"]
                ),
            ],
            axis=1,
        )
        df_headpoints["Label"] = df_headpoints["Label"].apply(
            lambda x: "HSP" if isinstance(x, np.ndarray) and x.size == 0 else x
        )  # change label from empty np array to "HSP" for headshape points
        nasion = (
            df_headpoints.loc[df_headpoints["Label"] == BS_LABEL_NAS, ["X", "Y", "Z"]]
            .mean()
            .to_numpy()
        )
        lpa = (
            df_headpoints.loc[df_headpoints["Label"] == BS_LABEL_LPA, ["X", "Y", "Z"]]
            .mean()
            .to_numpy()
        )
        rpa = (
            df_headpoints.loc[df_headpoints["Label"] == BS_LABEL_RPA, ["X", "Y", "Z"]]
            .mean()
            .to_numpy()
        )
        hsp = df_headpoints.loc[
            df_headpoints["Label"] == "HSP", ["X", "Y", "Z"]
        ].to_numpy()

        # Create MNE Montage
        dig = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
            hsp=hsp,
            coord_frame="unknown",
        )
        self.recording.dig = mne.channels.transform_to_head(dig)

    def evaluate_head_radius(self) -> None:
        """
        Reads head circumference (in meters) from file and calculates head radius.
        """
        with open(self.fpath_headcircum, "r") as f:
            content = json.load(f)
        self.recording.head_circum = float(content["Value"])
        self.recording.head_radius = self.recording.head_circum / (
            2 * np.pi
        )  # head circumference / 2π

    def _log_message(self, message: str | Exception):
        """
        Print given message in the default printing style.
        """
        if self.verbose:
            print("\n!!", message, "\n")
        else:
            pass
