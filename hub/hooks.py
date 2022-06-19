from typing import Callable, Optional
from uuid import uuid4

CREATE_DATASET_HOOKS = {}
LOAD_DATASET_HOOKS = {}
WRITE_DATASET_HOOKS = {}
READ_DATASET_HOOKS = {}
HOOK_EVENTS = {}


def _add_hook(event: str, hook: callable, id: Optional[str] = None):
    if id is None:
        id = str(uuid4())
    if id in HOOK_EVENTS:
        raise Exception(f"Hook with id {id} already exists.")
    globals()[f"{event}_HOOKS"][id] = hook
    HOOK_EVENTS[id] = event


def add_create_dataset_hook(hook: Callable, id: Optional[str] = None):
    _add_hook("CREATE_DATASET", hook, id)


def add_load_dataset_hook(hook: Callable, id: Optional[str] = None):
    _add_hook("LOAD_DATASET", hook, id)


def add_read_dataset_hook(hook: Callable, id: Optional[str] = None):
    _add_hook("READ_DATASET", hook, id)


def add_write_dataset_hook(hook: Callable, id: Optional[str] = None):
    _add_hook("WRITE_DATASET", hook, id)


def remove_hook(id: str):
    del globals()[f"_{HOOK_EVENTS.pop(id)}_HOOKS"][id]


def dataset_created(path: str):
    [f(path) for f in CREATE_DATASET_HOOKS.values()]


def dataset_loaded(path: str):
    [f(path) for f in LOAD_DATASET_HOOKS.values()]


def dataset_written(path: str):
    [f(path) for f in WRITE_DATASET_HOOKS.values()]


def dataset_read(path: str):
    [f(path) for f in READ_DATASET_HOOKS.values()]
