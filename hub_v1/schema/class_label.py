"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from typing import List, Tuple
from hub_v1.schema.features import Tensor


def _load_names_from_file(names_filepath):
    with open(names_filepath, "r") as f:
        return [name.strip() for name in f.read().split("\n") if name.strip()]


class ClassLabel(Tensor):
    """
    | Constructs a ClassLabel HubSchema.
    | Returns an integer representations of given classes. Preserves the names of classes to convert those back to strings if needed.
    | There are 3 ways to define a ClassLabel, which correspond to the 3 arguments:
           Note: In python2, the strings are encoded as utf-8.

    >>> import hub_v1
    >>> from hub_v1 import Dataset, schema
    >>> from hub_v1.schema import ClassLabel

    | 1. `num_classes`: create 0 to (num_classes-1) labels using ClassLabel(num_classes=`number of classes`)

    ----------
    >>> tag = "username/dataset"
    >>>
    >>> # Create dataset
    >>> ds=Dataset(
    >>>    tag,
    >>>    shape=(10,),
    >>>    schema = {
    >>>         "label_1": ClassLabel(num_classes=3),
    >>>    },
    >>> )
    >>>
    >>> ds["label_1",0] = 0
    >>> ds["label_1",1] = 1
    >>> ds["label_1",2] = 2
    >>>
    >>> ds.flush()
    >>>
    >>> # Load data
    >>> ds = Dataset(tag)
    >>>
    >>> print(ds["label_1"][0].compute(True))
    >>> print(ds["label_1"][1].compute(True))
    >>> print(ds["label_1"][2].compute(True))
    0
    1
    2


    | 2. `names`: a list of label strings. ClassLabel=(names=[`class1`,`class2`])

    ----------
    >>> tag = "username/dataset"
    >>>
    >>> # Define schema
    >>> my_schema = {
    >>>     "label_2": ClassLabel(names=['class1', 'class2', 'class3']),
    >>> }
    >>>
    >>> # Create dataset
    >>> ds=Dataset(
    >>>    tag,
    >>>    shape=(10,),
    >>>    schema = my_schema,
    >>> )
    >>>
    >>> ds.flush()
    >>>
    >>> # Load data
    >>> ds = Dataset(tag)

    | Note: ClassLabel HubSchema returnsan interger representation of classes.
    | Hence use `str2int()` and `int2str()` to load classes.

    >>> print(my_schema["label_2"].str2int("class1"))
    >>> print(my_schema["label_2"].int2str(0))
    0
    class1


    | 3. `names_file`: a file containing the list of labels. ClassLabel(names_file="/path/to/file/names.txt")

    Let's assume `names.txt` is located at `/dataset`:

    ----------
    >>> # Contents of "names.txt"
    welcome
    to
    hub


    >>> tag = "username/dataset"
    >>>
    >>> # Define Schema
    >>> my_schema = {
    >>>     "label_3": ClassLabel(names_file="/content/names.txt"),
    >>> }
    >>>
    # Create dataset
    >>> ds=Dataset(
    >>>    tag,
    >>>    shape=(10,),
    >>>    schema = my_schema,
    >>> )
    >>>
    >>> ds.flush()
    >>>
    >>> # Load data
    >>> ds = Dataset(tag)
    >>>
    >>> print(my_schema["label_3"].int2str(0))
    >>> print(my_schema["label_3"].int2str(1))
    >>> print(my_schema["label_3"].int2str(2))
    welcome
    to
    hub

    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        dtype="uint8",
        max_shape: Tuple[int, ...] = None,
        num_classes: int = None,
        names: List[str] = None,
        names_file: str = None,
        chunks=None,
        compressor="lz4",
    ):
        """

        Parameters
        ----------
        shape: tuple of ints or None
            The shape of classlabel.
            Will be () if only one classbabel corresponding to each sample.
            If N classlabels corresponding to each sample, shape should be (N,)
            If the number of classlabels for each sample vary from 0 to M. The shape should be set to (None,) and max_shape should be set to (M,)
            Defaults to ().
        max_shape : Tuple[int], optional
            Maximum shape of ClassLabel
        num_classes: `int`
            number of classes. All labels must be < num_classes.
        names: `list<str>`
            string names for the integer classes. The order in which the names are provided is kept.
        names_file: `str`
            path to a file with names for the integer classes, one per line.
        chunks : Tuple[int] | True, optional
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
        self.check_shape(shape)
        super().__init__(
            shape=shape,
            max_shape=max_shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
        )

        self._num_classes = None
        self._str2int = None
        self._int2str = None
        self._names = None

        if all(a is None for a in (num_classes, names, names_file)):
            return

        if sum(a is not None for a in (num_classes, names, names_file)) != 1:
            raise ValueError(
                "Only a single labeling argument of ClassLabel() should be provided."
            )

        if num_classes is not None:
            if isinstance(num_classes, int):
                self._num_classes = num_classes
        elif names is not None:
            self._names = names
            self.names = names
        else:
            self._names = _load_names_from_file(names_file)
            self.names = self._names

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
        return self.names[int_value]

    @property
    def num_classes(self):
        return self._num_classes

    def __str__(self):
        out = super().__str__()
        out = "ClassLabel" + out[6:-1]
        out = out + ", names=" + str(self._names) if self._names is not None else out
        out = (
            out + ", num_classes=" + str(self._num_classes)
            if self._num_classes is not None
            else out
        )
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()

    def check_shape(self, shape):
        if len(shape) not in [0, 1]:
            raise ValueError(
                "Wrong ClassLabel shape provided, should be of the format () or (None,) or (N,)"
            )
