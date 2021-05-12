import time


class Timer:
    def __init__(self, text):
        self._text = text

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        print(f"{self._text}: {time.time() - self._start}s")
