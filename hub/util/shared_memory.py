from typing import List

native_shared_memory = True
try:
    from multiprocessing.shared_memory import SharedMemory  # type: ignore
    from multiprocessing import resource_tracker  # type: ignore
except ImportError:
    native_shared_memory = False


def remove_shared_memory_from_resource_tracker():
    """Monkey-patch that fixes bug in Python SharedMemory
    More details at: https://bugs.python.org/issue38119
    """
    if not native_shared_memory:
        return

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
