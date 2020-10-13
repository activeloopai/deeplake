import typing


class Tensor:
    def __getitem__(self, slice_) -> typing.Union["Tensor", typing.Any]:
        pass

    def __setitem__(self, slice_, value):
        pass

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass
