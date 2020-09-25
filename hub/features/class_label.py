from typing import List

from hub.features.features import Tensor


def _load_names_from_file(names_filepath):
  with open(names_filepath, "r") as f:
    return [
        name.strip()
        for name in f.read().split("\n")
        if name.strip()  
    ]


class ClassLabel(Tensor):
    def __init__(self, num_classes: int = None,
                 names: List[str] = None, names_file: str = None
                ):
        super(ClassLabel, self).__init__(shape=(), dtype='int64')

        self._num_classes = None
        self._str2int = None
        self._int2str = None

        if all(a is None for a in (num_classes, names, names_file)):
            return

        if sum(a is not None for a in (num_classes, names, names_file)) != 1:
            raise ValueError(
                "Only a single argument of ClassLabel() should be provided.")

        if num_classes is not None:
            self._num_classes = num_classes
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
                ", new: {}".format(self._int2str, int2str))

        self._int2str = int2str
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError(
                "Some label names are duplicated. Each label name should be unique.")

        num_classes = len(self._str2int)
        if self._num_classes is None:
            self._num_classes = num_classes
        elif self._num_classes != num_classes:
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                "Got {} names VS {} num_classes".format(
                    num_classes, self._num_classes)
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

    def get_attribute_dict(self):
        """Return class attributes
        """
        return self.__dict__ 