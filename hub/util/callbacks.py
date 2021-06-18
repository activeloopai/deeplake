from typing import Callable, Any


class CallbackList(list):
    def __init__(self, callback: Callable, raw_list: list = []):
        """Acts exactly like a normal `list`, however when modifier methods are called (ie. `append`, `extend`, `__setitem__`),
            the provided `callback` method will be called immediately afterwards.

        Note: `raw_list` is recursively processed into callback compatible objects using `convert_to_callback_objects`.
            This also applies to all modifier methods.

        Args:
            callback (Callable): A function to be called after every update method call.
            raw_list (list): Starter list to be initialized with. Emulates `list(raw_list)`.
        """

        self.callback = callback

        callback_list = []
        for v in raw_list:
            callback_list.append(convert_to_callback_objects(v, self.callback))

        super().__init__(callback_list)

    def append(self, item):
        super().append(convert_to_callback_objects(item, self.callback))
        self.callback()

    def extend(self, items):
        super().extend(convert_to_callback_objects(items, self.callback))
        self.callback()

    def __setitem__(self, key, item):
        super().__setitem__(key, convert_to_callback_objects(item, self.callback))
        self.callback()


class CallbackDict(dict):
    def __init__(self, callback: Callable, raw_dict: dict = {}):
        """Acts exactly like a normal `dict`, however when modifier methods are called (ie. `update`, `__setitem__`),
            the provided `callback` method will be called immediately afterwards.

        Note: `raw_dict` is recursively processed into callback compatible objects using `convert_to_callback_objects`.
            This also applies to all modifier methods.

        Args:
            callback (Callable): A function to be called after every update method call.
            raw_dict (dict): Starter dictionary to be initialized with. Emulates `dict(raw_dict)`.
        """

        self.callback = callback

        callback_dict = {}
        for k, v in raw_dict.items():
            callback_dict[k] = convert_to_callback_objects(v, self.callback)

        super().__init__(callback_dict)

    def __setitem__(self, key, item):
        super().__setitem__(key, convert_to_callback_objects(item, self.callback))
        self.callback()

    def update(self, *args, **kwargs):
        converted = convert_to_callback_objects(dict(kwargs), self.callback)
        super().update(*args, **converted)
        self.callback()


def convert_to_callback_objects(value, callback: Callable):
    """Convert value into callback objects based on their type. For example, if `type(value) == list`,
    this will return a `CallbackList`.
    """

    if value in (CallbackList, CallbackDict):
        new_value = value(callback)
    elif isinstance(value, dict):
        new_value = CallbackDict(callback, value)
    elif isinstance(value, list):
        new_value = CallbackList(callback, value)
    else:
        new_value = value

    return new_value


def convert_from_callback_objects(value):
    """Convert value from callback objects into their subclass counterpart. For example, if `type(value) == CallbackList`,
    this will return a `list`.
    """

    if isinstance(value, CallbackDict):
        new_value = {k: convert_from_callback_objects(v) for k, v in value.items()}
    elif isinstance(value, CallbackList):
        new_value = [convert_from_callback_objects(item) for item in value]
    else:
        new_value = value

    return new_value
