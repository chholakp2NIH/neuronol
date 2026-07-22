import datetime as dt
import re
from pathlib import Path

import pandas as pd


class DataImporter:
    fpath_import: Path
    df_imotions: pd.DataFrame
    recording_dt: dt.datetime
    df_eeg: pd.DataFrame

    # Given
    possible_unwanted_columns = (
        ["Row", "SourceStimuliName", "SampleNumber"]
        + ["Combined Event Source", "EventSource"]
        + ["Aux%d" % w for w in range(1, 9)]
        + ["Channel %d" % w for w in range(33, 41)]
    )

    def __init__(self, fpath_import) -> None:
        self.fpath_import = fpath_import

    def read_imotions_csv_as_df(self) -> None:
        """
        Read input iMotions CSV data (`self.fpath_import`) as a DataFrame.
        """
        self.df_imotions = pd.read_csv(self.fpath_import, low_memory=False)
        return None

    def get_recording_datetime_from_imotions_df(self):
        """
        Find the date/time of the iMotions recording from the `self.df_imotions`
        """
        inx_rectime = self.df_imotions.index[
            self.df_imotions.iloc[:, 0] == "#Recording time"
        ].to_list()[0]
        date_str = ""
        time_str = ""
        datetime_str = ""
        for data in self.df_imotions.iloc[inx_rectime].astype(str):
            if "Date" in data:
                m = re.search(r"^Date: (\d{4}-\d{2}-\d{2})$", data)
                if m is not None:
                    date_str = m.group(1)
            elif "Time" in data:
                m = re.search(r"^Time: (.+) .+$", data)
                if m is not None:
                    time_str = m.group(1)
        if (date_str != "") and (time_str != ""):
            datetime_str = date_str + " " + time_str
        else:
            raise Exception("\n(!!) Recording date and time strings not found\n")
        self.recording_dt = dt.datetime.fromisoformat(datetime_str)
        return None

    def drop_header_info_from_df(self):
        """
        Drop additional info (CSV header info) from imported `self.df_imotions`
        """
        # Drop general info at top
        inx_first_timepoint = self.df_imotions.index[
            self.df_imotions.iloc[:, 0] == "1"
        ].to_list()[0]
        inx_colnames = inx_first_timepoint - 1
        self.df_imotions.drop(self.df_imotions.head(inx_colnames).index, inplace=True)

        # Change column names
        colnames = self.df_imotions.iloc[0].values
        self.df_imotions = self.df_imotions[1:]
        self.df_imotions.columns = colnames
        return None

    def extract_eeg_from_imotions_data(self, possible_unwanted_columns):
        """
        Extract EEG data from iMotions dataframe. Returns EEG data in Volts.
        """

        df_eeg = df_imotions.drop(
            df_imotions.index[df_imotions["Fp1"].isna()]
        )  # drop rows without EEG data
        df_eeg.dropna(
            axis=1, how="all", inplace=True
        )  # drop columns that don't have any data left
        available_unwanted_columns = [
            c for c in possible_unwanted_columns if c in df_eeg.columns
        ]
        df_eeg.drop(
            columns=available_unwanted_columns, inplace=True
        )  # drop avail unwanted cols
        df_eeg = df_eeg.astype("float")
        df_eeg.iloc[:, 1:] /= 1e6  # convert EEG signal unit from μV to V
        df_eeg.reset_index(drop=True, inplace=True)

        return df_eeg
