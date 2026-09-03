FROM python:3.14

WORKDIR /app

# Install neuronol and its dependencies
COPY pyproject.toml .
COPY neuronol ./neuronol
RUN python -m pip install --no-cache-dir .

# Copy the paired python example script
COPY examples/data_import/eegacamp_import_with_watcher.py ./examples/data_import/eegacamp_import_with_watcher.py

# Change working dir to the example script dir and run script
WORKDIR /app/examples/data_import
CMD ["python", "./eegacamp_import_with_watcher.py"]
