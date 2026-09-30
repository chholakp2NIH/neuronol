# Automatic EEGAcamp Data Import

This example uses `DataWatcher` and `DBManager` to automatically import
completed EEGAcamp recordings and track imported recordings in a SQLite
database.

## Run the automatic data importer

Start the long-running watcher:

```bash
docker compose up -d
```

The importer watches for newly completed recordings and imports them
automatically.

## Re-import all data

Rebuild the database and re-import all completed recordings:

```bash
docker compose run --rm auto-data-importer \
    python -u -X faulthandler eegacamp_import_data.py --recreate-all
```

This deletes the existing database, recreates it, imports all recordings
containing a `.data_collection_file`, and exits.
