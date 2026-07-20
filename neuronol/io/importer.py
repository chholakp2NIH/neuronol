import os
from pathlib import Path

import pandas as pd


class DataImporter:
    fpath_import: Path
    df: pd.DataFrame

    def __init__(self, fpath_import) -> None:
        self.fpath_import = fpath_import

    def read_data_as_df(self) -> None:
        """
        Read input data (`self.fpath_import`) as a DataFrame.
        """
        self.df = pd.read_csv(self.fpath_import, low_memory=False)
        return None
