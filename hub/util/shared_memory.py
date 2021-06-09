from typing import List

try:
    from multiprocessing.shared_memory import SharedMemory  # type: ignore
    from multiprocessing import resource_tracker  # type: ignore
except ImportError:
    pass


def remove_shared_memory_from_resource_tracker():
    """Monkey-patch that fixes bug in Python SharedMemory
    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    resource_tracker.register = fix_register
    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


def clear_shared_memory(shared_memory_names: List[str]):
    """Checks if an existing SharedMemory exists for any chunk in chunk_names and clears it"""
    for chunk_name in shared_memory_names:
        try:
            shm = SharedMemory(name=chunk_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
