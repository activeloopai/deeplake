from deeplake.client.log import configure_logger
from logging import StreamHandler
from itertools import cycle
from functools import wraps

import deeplake
import contextlib
import threading
import logging
import time
import sys

ACTIVE_SPINNER = None


class DummyFile:
    def __init__(self, file, spinner):
        self.spinner = spinner
        self.file = file

    def write(self, text):
        if len(text.strip()) > 0:
            if self.spinner._hide_event.is_set():
                self.file.write(text)
            else:
                with self.spinner._stderr_lock:
                    self.spinner._clear_line()
                    self.file.write(f"{text}\n")

    def __getattr__(self, attr):
        return getattr(self.file, attr)


@contextlib.contextmanager
def run_spinner(spinner):
    global ACTIVE_SPINNER
    try:
        if not isinstance(sys.stderr, DummyFile) and deeplake.constants.SPINNER_ENABLED:
            spinner.start()
            spinner_started = True
            save_stdout = sys.stdout
            save_stderr = sys.stderr
            sys.stdout = DummyFile(sys.stdout, spinner)
            sys.stderr = DummyFile(sys.stderr, spinner)
            # configure logger to use new stdout
            logger = logging.getLogger("deeplake")
            save_handlers = list(logger.handlers)
            logger.handlers.clear()
            logger.addHandler(StreamHandler(stream=sys.stdout))
            ACTIVE_SPINNER = spinner
        else:
            # another spinner active
            # or tests running
            spinner_started = False
        yield
    finally:
        if spinner_started:
            spinner.stop()
            sys.stdout = save_stdout
            sys.stderr = save_stderr
            logger.handlers = save_handlers
            ACTIVE_SPINNER = None


class Spinner(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()
        self._hide_event = threading.Event()
        self._cur_line_len = 0
        self.daemon = True
        self._stderr_lock = threading.Lock()
        self.file = sys.stderr

    def run(self):
        try:
            time.sleep(deeplake.constants.SPINNER_START_DELAY)
            frames = cycle("/-\\|")
            if not self._hide_event.is_set() and not self._stop_event.is_set():
                self._hide_cursor()
            while not self._stop_event.is_set():
                if self._hide_event.is_set():
                    time.sleep(0.1)
                    continue
                with self._stderr_lock:
                    self._clear_line()
                    self.file.write(next(frames))
                    self.file.flush()
                    self._cur_line_len = 1

                self._stop_event.wait(0.1)
        except ValueError:
            # I/O operation on closed file
            pass

    def hide(self):
        if not self._hide_event.is_set():
            with self._stderr_lock:
                self._hide_event.set()
                self._clear_line()
                self.file.flush()
                self._show_cursor()

    def show(self):
        if self._hide_event.is_set():
            with self._stderr_lock:
                self._hide_event.clear()
                self.file.write("\n")
                self._hide_cursor()

    def stop(self):
        try:
            self._stop_event.set()
            if not self._hide_event.is_set():
                self._clear_line()
            self._show_cursor()
        except ValueError:
            # I/O operation on closed file
            pass

    def _clear_line(self):
        if not self.file.closed:
            if self.file.isatty():
                # ANSI Control Sequence EL does not work in Jupyter
                self.file.write("\r\033[K")
            else:
                fill = " " * self._cur_line_len
                self.file.write(f"\r{fill}\r")
            self._cur_line_len = 0

    def _hide_cursor(self):
        if not self.file.closed and self.file.isatty():
            # ANSI Control Sequence DECTCEM 1 does not work in Jupyter
            self.file.write("\033[?25l")
            self.file.flush()

    def _show_cursor(self):
        if not self.file.closed and self.file.isatty():
            # ANSI Control Sequence DECTCEM 2 does not work in Jupyter
            self.file.write("\033[?25h")
            self.file.flush()


def spinner(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if kwargs.pop("spinner", True) and kwargs.get("verbose") in (None, True):
            spinner = Spinner()

            with run_spinner(spinner):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return inner
