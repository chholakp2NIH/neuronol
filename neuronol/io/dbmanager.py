import sqlite3
from pathlib import Path
from typing import Literal

import pandas as pd

from neuronol.constants import DEFAULT_DB_INIT_SCRIPT


class DBManager:

    def __init__(
        self,
        path: Path | str,
        initialize_db: bool = False,
        db_init_script: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True, exist_ok=True
        )  # ensure parent dir to db exists, otherwise create
        self.conn = self.connect()

        # Database initialization
        if db_init_script is not None:
            self.db_init_script = db_init_script
        else:
            self.db_init_script = DEFAULT_DB_INIT_SCRIPT
        if initialize_db:
            self.init_db()

    # Connect to db file
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # Initialize db
    def init_db(self):
        cur = self.conn.cursor()
        cur.executescript(self.db_init_script)
        self.conn.commit()

    # Run SQL query on db
    def run_sql_query(self, query: str):
        """
        Run SQL `query` on db.
        """
        cur = self.conn.cursor()
        cur.execute(query)
        # return [r[0:] for r in cur.fetchall()]
        return cur.fetchall()

    # Read col values from table
    def read_col_values_from_table(
        self, table_name, col_name, return_id=None, id_col="ID"
    ):
        """
        Read all values under `col_name` from `table_name`.
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        if return_id and return_id is not None:
            return [(r[id_col], r[col_name]) for r in cur.fetchall()]
        return [r[col_name] for r in cur.fetchall()]

    # Write col values to table
    def add_row_to_table(
        self,
        table_name,
        col_names: tuple[str],
        values: list[tuple],
    ):
        """
        Add row(s) of data under list column names to table.
        """
        cur = self.conn.cursor()
        placeholders = ", ".join(["?"] * len(col_names))
        columns = ", ".join(col_names)
        script = f"""
        INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
        """
        cur.executemany(script, values)
        self.conn.commit()

    # Convert table to df
    def import_table_as_df(self, table_name: str):
        """
        Import db table as a DataFrame.
        """
        return pd.read_sql_query(f"SELECT * FROM {table_name};", self.conn)

    # Convert df to table
    def export_df_as_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = False,
        **kwargs,
    ):
        """
        Export pandas DataFrame as db table.
        """
        return df.to_sql(
            table_name, self.conn, if_exists=if_exists, index=index, **kwargs
        )
