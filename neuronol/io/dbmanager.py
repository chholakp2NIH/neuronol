import sqlite3
from pathlib import Path

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
    def read_col_from_table(self, table_name, col_name, return_id=None):
        """
        Read all values under `col_name` from `table_name`.
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        if return_id and return_id is not None:
            return [(r["ID"], r[col_name]) for r in cur.fetchall()]
        return [r[col_name] for r in cur.fetchall()]

    # Convert table to df
    def export_table_as_df(self, table_name):
        """
        Export db table as a DataFrame.
        """
        return pd.read_sql_query(f"SELECT * FROM {table_name};", self.conn)

    # # Create new table from df
    # def create_table_from_df(self, df: pd.DataFrame, table_name):
    #     """
    #     Create DB table from DataFrame `df`.
    #     """
    #     # Create empty table
    #     col_names = tuple(w for w in df.columns)
    #     col_names = ("ID",) + col_names
    #     cur = self.conn.cursor()
    #     cur.executescript(f"CREATE TABLE {table_name} {col_names};")
    #     # Get col values row-by-row and insert into table
    #     for ind, row in df.iterrows():
    #         cur.execute(
    #             f"INSERT INTO {table_name} {col_names} VALUES ({', '.join(['?'] * len(col_names))})",
    #             tuple([ind] + row.to_list()),
    #         )
    #     # Commit changes to db
    #     self.conn.commit()
