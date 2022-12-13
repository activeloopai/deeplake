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
        elif self.reason == "update":
            raise InvalidViewException(
                "This dataset view was invalidated because changes were made at the HEAD node after creation of this view."
            )

    __getattr__ = __getitem__
