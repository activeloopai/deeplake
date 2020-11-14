from typing import List
from hub.features.features import Tensor


def _load_names_from_file(names_filepath):
    with open(names_filepath, "r") as f:
        return [name.strip() for name in f.read().split("\n") if name.strip()]


class ClassLabel(Tensor):
    """`HubFeature` for integer class labels."""

    def __init__(
        self,
        num_classes: int = None,
        names: List[str] = None,
        names_file: str = None,
        chunks=None,
        compressor="lz4",
    ):
        """| Constructs a ClassLabel HubFeature.
        | There are 3 ways to define a ClassLabel, which correspond to the 3 arguments:
        | * `num_classes`: create 0 to (num_classes-1) labels
        | * `names`: a list of label strings
        | * `names_file`: a file containing the list of labels.
        Note: In python2, the strings are encoded as utf-8.

        | Usage:
        ----------
        >>> class_label_tensor = ClassLabel(num_classes=10)
        >>> class_label_tensor = ClassLabel(names=['class1', 'class2', 'class3', ...])
        >>> class_label_tensor = ClassLabel(names_file='/path/to/file/with/names')

        Parameters
        ----------
        num_classes: `int`
            number of classes. All labels must be < num_classes.
        names: `list<str>`
            string names for the integer classes. The order in which the names are provided is kept.
        names_file: `str`
            path to a file with names for the integer classes, one per line.
        max_shape : Tuple[int]
            Maximum shape of tensor shape if tensor is dynamic
        chunks : Tuple[int] | True
            Describes how to split tensor dimensions into chunks (files) to store them efficiently.
            It is anticipated that each file should be ~16MB.
            Sample Count is also in the list of tensor's dimensions (first dimension)
            If default value is chosen, automatically detects how to split into chunks

        | Note: Only num_classes argument can be filled, providing number of classes,
              names or names file

        Raises
        ----------
        ValueError: If more than one argument is provided
        """
        super().__init__(
            shape=(),
            dtype="int64",
            chunks=chunks,
            compressor=compressor,
        )

        self._num_classes = None
        self._str2int = None
        self._int2str = None

        if all(a is None for a in (num_classes, names, names_file)):
            return

        if sum(a is not None for a in (num_classes, names, names_file)) != 1:
            raise ValueError(
                "Only a single labeling argument of ClassLabel() should be provided."
            )

        if num_classes is not None:
            if isinstance(num_classes, int):
                self._num_classes = num_classes
            # elif isinstance(num_classes, List):
            #     names = num_classes
            # elif isinstance(num_classes, str):
            #     names_file = num_classes
        elif names is not None:
            self.names = names
        else:
            self.names = _load_names_from_file(names_file)

    @property
    def names(self):
        if not self._int2str:
            return [str(i) for i in range(self._num_classes)]
        return list(self._int2str)

    @names.setter
    def names(self, new_names):
        int2str = [name for name in new_names]
        if self._int2str is not None and self._int2str != int2str:
            raise ValueError(
                "Trying to overwrite already defined ClassLabel names. Previous: {} "
                ", new: {}".format(self._int2str, int2str)
            )

        self._int2str = int2str
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError(
                "Some label names are duplicated. Each label name should be unique."
            )

        num_classes = len(self._str2int)
        if self._num_classes is None:
            self._num_classes = num_classes
        elif self._num_classes != num_classes:
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                "Got {} names VS {} num_classes".format(num_classes, self._num_classes)
            )

    def str2int(self, str_value: str):
        """Conversion class name string => integer."""
        if self._str2int:
            return self._str2int[str_value]

        failed_parse = False
        try:
            int_value = int(str_value)
        except ValueError:
            failed_parse = True
        if failed_parse or not 0 <= int_value < self._num_classes:
            raise ValueError("Invalid string class label %s" % str_value)
        return int_value

    def int2str(self, int_value: int):
        """Conversion integer => class name string."""
        if self._int2str:
            return self._int2str[int_value]

    @property
    def num_classes(self):
        return self._num_classes

    def get_attr_dict(self):
        """Return class attributes."""
        return self.__dict__
