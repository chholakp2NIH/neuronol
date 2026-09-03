FROM python:3.14

WORKDIR /app

# Install neuronol and its dependencies
COPY pyproject.toml .
COPY neuronol ./neuronol
RUN python -m pip install --no-cache-dir .

# Copy the paired python example script
COPY examples/data_import/watcher+db/eegacamp_import_with_watcher_and_db.py ./examples/data_import/watcher+db/eegacamp_import_with_watcher_and_db.py
COPY examples/data_import/_eegacamp.py ./examples/data_import/_eegacamp.py

# Run script
WORKDIR /app/examples/data_import/watcher+db
# CMD ["python", "-u", "./eegacamp_import_with_watcher_and_db.py"]
CMD ["python", "-u", "-X", "faulthandler", "eegacamp_import_with_watcher_and_db.py"]
