from pathlib import Path
from queue import Queue

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from neuronol.constants import WATCHER_DATA_COPY_COMPLETED


class DataEventHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self.queue = Queue()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            fpath = Path(str(event.src_path))
            if fpath.name == WATCHER_DATA_COPY_COMPLETED:
                self.queue.put(fpath.parent)


class DataWatcher:
    def __init__(self, data_dir) -> None:
        self.observer = Observer()
        self.event_handler = DataEventHandler()
        self.observer.schedule(self.event_handler, data_dir, recursive=True)
