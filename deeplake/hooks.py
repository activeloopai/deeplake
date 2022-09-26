from typing import Callable, Dict, Optional
from uuid import uuid4

CREATE_DATASET_HOOKS: Dict[str, Callable] = {}
LOAD_DATASET_HOOKS: Dict[str, Callable] = {}
WRITE_DATASET_HOOKS: Dict[str, Callable] = {}
READ_DATASET_HOOKS: Dict[str, Callable] = {}
COMMIT_DATASET_HOOKS: Dict[str, Callable] = {}
HOOK_EVENTS: Dict[str, str] = {}


def _add_hook(event: str, hook: Callable, id: Optional[str] = None):
    if id is None:
        id = str(uuid4())
    if id in HOOK_EVENTS:
        return
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


def add_commit_dataset_hook(hook: Callable, id: Optional[str] = None):
    _add_hook("COMMIT_DATASET", hook, id)


def remove_hook(id: str):
    del globals()[f"_{HOOK_EVENTS.pop(id)}_HOOKS"][id]


def dataset_created(ds):
    [f(ds) for f in CREATE_DATASET_HOOKS.values()]


def dataset_loaded(ds):
    [f(ds) for f in LOAD_DATASET_HOOKS.values()]


def dataset_written(ds):
    [f(ds) for f in WRITE_DATASET_HOOKS.values()]


def dataset_read(ds):
    [f(ds) for f in READ_DATASET_HOOKS.values()]


def dataset_committed(ds):
    [f(ds) for f in COMMIT_DATASET_HOOKS.values()]
