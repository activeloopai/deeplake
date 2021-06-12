from typing import Callable, Any


class CallbackList(list):
    def __init__(self, callback: Callable, raw_list: list=[]):
        self.callback = callback

        # TODO: generalize callbacks to a list of callback functions

        # TODO: handle recursive
        callback_list = []
        for v in raw_list:
            callback_list.append(convert_to_callback_classes(v, self.callback))

        super().__init__(callback_list)

    def append(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().append(*args)
        self.callback()

    def extend(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().extend(*args)
        self.callback()

    def __setitem__(self, *args):
        super().__setitem__(*args)
        self.callback()


class CallbackDict(dict):
    def __init__(self, callback: Callable, raw_dict: dict={}):
        self.callback = callback

        # TODO: handle recursive
        callback_dict = {}
        for k, v in raw_dict.items():
            callback_dict[k] = convert_to_callback_classes(v, self.callback)
        
        super().__init__(callback_dict)

    def __setitem__(self, *args):
        # TODO: only support list/dictionary objects (and parse them to be CallbackDicts/CallbackLists)
        super().__setitem__(*args)
        self.callback()

    def update(self, *args):
        super().update(*args)
        self.callback()

def convert_to_callback_classes(value: Any, callback: Callable):
    # TODO: explain what's going on here

    # TODO: check if value is supported `type` (we should only support what json supports)

    if value in (CallbackList, CallbackDict):
        new_value = value(callback)
    elif isinstance(value, dict):
        new_value = CallbackDict(callback, value)
    elif isinstance(value, list):
        new_value = CallbackList(callback, value)
    else:
        new_value = value

    return new_value


def convert_from_callback_classes(value: Any):
    # TODO: explain what's going on here

    if isinstance(value, CallbackDict):
        new_value = dict(value)
    elif isinstance(value, CallbackList):
        new_value = list(value)
    else:
        new_value = value

    return new_value