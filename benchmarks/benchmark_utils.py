import datetime
import time
import os
import json
import atexit
import itertools
import builtins

REPORTS_DIR = ".reports"
if not os.path.isdir(REPORTS_DIR):
    os.mkdir(REPORTS_DIR)


CURRENT_RUN = None


def _exit_hook():
    if CURRENT_RUN:
        CURRENT_RUN.write()
        CURRENT_RUN.summary()


atexit.register(_exit_hook)


class Run(object):
    def __init__(self, path=None):
        if path:
            self.path = path
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            now = datetime.datetime.now()
            fname = (
                f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
            )
            fname = os.path.join(REPORTS_DIR, fname)
            i = 2
            while os.path.isfile(fname + ".json"):
                fname = fname + str(i)
                i += 1
            fname = fname + ".json"
            self.path = fname
            self.data = {}

    def report(self, path, config, start, end):
        path_elems = path.split(".")
        root = path_elems[0]
        if root not in self.data:
            self.data[root] = {"inner": {}, "iterations": [], "path": root}
        d = self.data[root]
        curr_path = root
        for e in path_elems[1:]:
            curr_path += "." + e
            print(d)
            if e not in d["inner"]:
                d_next = {"inner": {}, "iterations": [], "path": curr_path}
                d["inner"][e] = d_next
                d = d_next
        interval = end - start
        d["iterations"].append(
            {"config": config, "start": start, "end": end, "interval": interval}
        )
        print(f"Report received: {path} - Time: {interval} seconds - Config: {config}")

    def write(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def _get_report(self, path):
        d = self.data
        path_elems = path.split(".")
        root = path_elems[0]
        d = self.data[root]
        for e in path_elems[1:]:
            d = d["inner"][e]
        return d

    def _summary(self, path):
        depth = path.count(".'")
        r = self._get_report(path)
        orig_print = builtins.print
        print = lambda x: orig_print(" " * depth + x)
        print(f"===== Summary: {path} =====")
        num_iters = len(r["iterations"])
        if num_iters:
            print(f"{num_iters} Iterations")
            for i, it in enumerate(r["iterations"]):
                print(
                    f"Iteration {i + 1}: {int(it['interval'])} seconds - {it['config']}"
                )
            times = [it["interval"] for it in r["iterations"]]
            print(f"Mean time: {sum(times) / len(times)}")
            print(f"Slowest time: {max(times)}")
            print(
                f"Slowest cofig: {max(r['iterations'], key=lambda it: it['interval'])['config']}"
            )
            print(f"fastest time: {min(times)}")
            print(
                f"fastest config: {min(r['iterations'], key=lambda it: it['interval'])['config']}"
            )
        for inner in r["inner"].values():
            self._summary(inner["path"])

    def summary(self):
        for k in self.data:
            self._summary(k)


def report(path, config, start, end):
    global CURRENT_RUN
    if CURRENT_RUN is None:
        CURRENT_RUN = Run()
    CURRENT_RUN.report(path, config, start, end)


_timers = {}


def _get_timer_children(path):
    return [k for k in _timers if k.startswith(path + ".")]


def _get_timer_parent(path):
    return os.path.splitext(path)[0]


def timer(path="", config=None, callback=None):
    if path in _timers:
        tmr = _timers[path]
        if tmr.active:
            raise Exception()
        tmr.config = {} or config
        return tmr
    tmr = Timer(path, config, callback)
    return tmr


class Timer(object):
    def __init__(self, path=None, config=None, callback=None):
        if not path:
            path = "__main__"
        if path in _timers:
            tmr = _timers[path]
            if tmr.active:
                raise Exception()
        _timers[path] = self
        self.path = path
        self.config = config or {}
        self.callback = callback
        self.active = False

    def __enter__(self):
        self.start = time.time()
        self.active = True

    def __exit__(self, *args):
        self.active = False
        end = time.time()
        report(self.path, self.config, start=self.start, end=end)
        if self.callback:
            callback(path, config, start=self.start, end=end)


def parametrize(**kwargs):
    def f(f_orig):
        def f_parametrizded():
            for perm in itertools.product(*(kwargs.values())):
                next_kwargs = dict(zip(kwargs.keys(), perm))
                print(f"Running {f_orig.__name__} with arguments {next_kwargs}")
                f_orig(**next_kwargs)

        return f_parametrizded

    return f
