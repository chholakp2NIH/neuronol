import os
import sqlite3

import pandas as pd

from neuronol.constants import DEFAULT_DB_INIT_SCRIPT
from neuronol.io.dbmanager import DBManager


def test_dbmanager_init(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    # Default configs
    db_manager = DBManager(fpath_db)
    assert isinstance(db_manager.path, os.PathLike)
    assert isinstance(db_manager.conn, sqlite3.Connection)
    assert db_manager.path == fpath_db
    assert db_manager.db_init_script == DEFAULT_DB_INIT_SCRIPT
    # Set db init script
    db_manager = DBManager(fpath_db, db_init_script="xyz")
    assert db_manager.db_init_script == "xyz"


def test_connect(tmp_path):
    db_manager = DBManager(tmp_path / "tmp.db")
    conn = db_manager.connect()
    assert isinstance(conn, sqlite3.Connection)
    assert conn.row_factory == sqlite3.Row


def test_init_db(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    db_manager = DBManager(fpath_db)
    db_manager.init_db()
    assert fpath_db.stat().st_size > 0


def test_run_sql_query(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    db_init_script = (
        "CREATE TABLE participants (sub_id PRIMARY_KEY);"
        + "INSERT INTO participants (sub_id) VALUES ('sub-01');"
    )
    db_manager = DBManager(
        fpath_db,
        initialize_db=True,
        db_init_script=db_init_script,
    )
    rows = db_manager.run_sql_query("SELECT * FROM participants;")
    assert all([isinstance(w, sqlite3.Row) for w in rows])
    assert len(rows) == 1


def test_read_col_from_table(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    db_init_script = (
        "CREATE TABLE participants (ID PRIMARY_KEY, sub_id, score);"
        + "INSERT INTO participants (ID, sub_id, score) VALUES "
        + "(1, 'sub-01', 100), "
        + "(2, 'sub-02', 80), "
        + "(3, 'sub-03', 90);"
    )
    db_manager = DBManager(
        fpath_db,
        initialize_db=True,
        db_init_script=db_init_script,
    )
    # Test default config
    vals = db_manager.read_col_values_from_table("participants", "score")
    assert vals == [100, 80, 90]
    # Test return_id with default ID column
    vals_indexed = db_manager.read_col_values_from_table(
        "participants", "score", return_id=True
    )
    ids = [w[0] for w in vals_indexed]
    assert ids == [1, 2, 3]
    vals = [w[1] for w in vals_indexed]
    assert vals == [100, 80, 90]
    # Test return_id with custom ID column
    vals_indexed = db_manager.read_col_values_from_table(
        "participants", "score", return_id=True, id_col="sub_id"
    )
    ids = [w[0] for w in vals_indexed]
    assert ids == ["sub-01", "sub-02", "sub-03"]
    vals = [w[1] for w in vals_indexed]
    assert vals == [100, 80, 90]


def test_import_table_as_df(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    db_init_script = (
        "CREATE TABLE participants (sub_id PRIMARY_KEY);"
        + "INSERT INTO participants (sub_id) VALUES ('sub-01');"
    )
    db_manager = DBManager(
        fpath_db,
        initialize_db=True,
        db_init_script=db_init_script,
    )
    df = db_manager.import_table_as_df("participants")
    assert isinstance(df, pd.DataFrame)
    assert df.columns == ["sub_id"]
    assert df["sub_id"].to_list() == ["sub-01"]


def test_export_df_as_table(tmp_path):
    fpath_db = tmp_path / "tmp.db"
    db_manager = DBManager(fpath_db)
    df = pd.DataFrame({"sub_id": ["sub-01", "sub-02"], "score": [100, 90]})
    db_manager.export_df_as_table(df, "SubjectScores")
    df_read = db_manager.import_table_as_df("SubjectScores")
    assert all(df == df_read)


def test_add_rows_to_table(tmp_path):
    db_init_script = "CREATE TABLE participants (sub_id PRIMARY_KEY, score);"
    # Add single row
    db_manager = DBManager(
        tmp_path / "db_single_row.db", initialize_db=True, db_init_script=db_init_script
    )
    db_manager.add_row_to_table(
        "participants",
        ("sub_id", "score"),
        values=[("sub-01", 100)],
    )
    sub_ids = db_manager.read_col_values_from_table("participants", "sub_id")
    scores = db_manager.read_col_values_from_table("participants", "score")
    assert sub_ids == ["sub-01"]
    assert scores == [100]
    # Add multiple rows
    db_manager = DBManager(
        tmp_path / "db_multiple_rows.db",
        initialize_db=True,
        db_init_script=db_init_script,
    )
    db_manager.add_row_to_table(
        "participants",
        ("sub_id", "score"),
        values=[("sub-01", 100), ("sub-02", 80)],
    )
    sub_ids = db_manager.read_col_values_from_table("participants", "sub_id")
    scores = db_manager.read_col_values_from_table("participants", "score")
    assert sub_ids == ["sub-01", "sub-02"]
    assert scores == [100, 80]
