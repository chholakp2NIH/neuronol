import os
import sqlite3

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
    db_manager = DBManager(
        fpath_db,
        initialize_db=True,
        db_init_script="CREATE TABLE participants (sub_id PRIMARY_KEY);",
    )
    query = "SELECT * FROM participants;"
    rows = db_manager.run_sql_query(query)
    assert rows is not None
    assert len(rows) == 0
