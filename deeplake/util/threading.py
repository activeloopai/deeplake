import ctypes


def terminate_thread(thread):
    """Terminates a python thread from another thread."""

    if not thread.is_alive():
        return

    exc = ctypes.py_object(Exception)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
