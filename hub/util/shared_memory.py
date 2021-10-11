import os
import warnings
import subprocess
from sys import platform
from hub.constants import GB, MB


try:
    from multiprocessing.shared_memory import SharedMemory  # type: ignore
    from multiprocessing import resource_tracker  # type: ignore
    from hub.core.storage import SharedMemoryProvider
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


def clear_shared_memory():
    """Clears existing Hub shared memory on linux systems."""
    shm_storage = SharedMemoryProvider()
    file_names = os.listdir("/dev/shm")
    hub_files = [
        name for name in file_names if name.startswith("al_") or name.startswith("tr_")
    ]
    for file_name in hub_files:
        try:
            del shm_storage[file_name]
        except:
            pass


def get_final_buffer_size(buffer_size_in_mb):
    """Checks the size of the buffer to see if enough shared memory is available. Returns the final size of buffer in bytes."""
    buffer_size_in_bytes = buffer_size_in_mb * MB

    if "linux" in platform:
        clear_shared_memory()
        output = subprocess.check_output("df -H /dev/shm", shell=True)
        shm_size = float(output.split(b"\n")[1].split()[3][:-1])
        shm_size_in_bytes = shm_size * GB
        shm_size_in_mb = shm_size * 1000
        if shm_size_in_bytes < buffer_size_in_bytes:
            warnings.warn(
                UserWarning(
                    f"The available shared memory size({shm_size_in_mb} MB) is less than the specified buffer size({buffer_size_in_mb} MB). "
                    f"Setting the buffer size to {0.95 * shm_size_in_mb} MB."
                )
            )
            return 0.95 * shm_size_in_bytes
    return buffer_size_in_bytes
