import numpy as np
from typing import List, Optional
from terminaltables import AsciiTable


class DatasetInfo:
    def __init__(self, mode: bool, classes: Optional[List[str]] = None):
        """
        Initialize the DatasetInfo class.

        Args:
            mode (bool): A boolean value indicating if the dataset is in test mode.
            classes (Optional[List[str]]): A list of class names, or None if class names are not provided.
        """
        self.test_mode = mode
        self.CLASSES = classes

    def count_instances(self) -> np.ndarray:
        """
        Count instances of each class in the dataset.

        Return:
            np.ndarray: A NumPy array containing instance counts for each class.
        """
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        for idx in range(len(self)):
            label = self.get_ann_info(idx)["labels"]
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                instance_count[unique] += counts
            else:
                instance_count[-1] += 1
        return instance_count

    def format_table_data(self, instance_count: np.ndarray) -> List[List[str]]:
        """
        Format the instance count data for display as a table.

        Args:
            instance_count (np.ndarray): A NumPy array containing instance counts for each class.

        Return:
            List[List[str]]: A list of lists containing formatted table data.
        """
        table_data = [["category", "count"] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f"{cls} [{self.CLASSES[cls]}]", f"{count}"]
            else:
                row_data += ["-1 background", f"{count}"]

            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []

        if len(row_data) >= 2:
            if row_data[-1] == "0":
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        return table_data

    def __str__(self) -> str:
        """
        Generate a string representation of the dataset information.

        Return:
            str: A formatted string describing the dataset and instance counts.
        """
        dataset_type = "Test" if self.test_mode else "Train"
        result = (
            f"\n{self.__class__.__name__} {dataset_type} dataset "
            f"with number of images {len(self)}, "
            f"and instance counts: \n"
        )
        if self.CLASSES is None:
            result += "Category names are not provided. \n"
            return result

        instance_count = self.count_instances()
        table_data = self.format_table_data
        table = AsciiTable(table_data)
        result += table.table

        return result
