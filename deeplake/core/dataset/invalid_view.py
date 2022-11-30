from deeplake.util.exceptions import InvalidViewException

import deeplake


class InvalidView:
    def __init__(self, reason):
        self.reason = reason

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, deeplake.Dataset)

    def __getitem__(self, item):
        if self.reason == "checkout":
            raise InvalidViewException(
                "This dataset view was invalidated because the base dataset was checked out to a different commit."
            )

    __getattr__ = __getitem__
