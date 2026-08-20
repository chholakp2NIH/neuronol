import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from neuronol.constants import WATCHER_DATA_COPY_COMPLETED


class DataEventHandler(FileSystemEventHandler):
    def __init__(self, callback) -> None:
        self.callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        print(event)
        if not event.is_directory:
            fpath = Path(str(event.src_path))
            if fpath.name == WATCHER_DATA_COPY_COMPLETED:
                print(fpath.name)
                self.callback(fpath.parent)


class DataWatcher:
    def __init__(self, callback, data_dir) -> None:
        self.observer = Observer()
        self.event_handler = DataEventHandler(callback)
        self.observer.schedule(self.event_handler, data_dir, recursive=True)

    def scout(self):
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            self.observer.stop()
            self.observer.join()
