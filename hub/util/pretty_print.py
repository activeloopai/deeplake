def max_array_length(arrMax, arrToCompare):  # helper for __str__
    for i in range(len(arrMax)):
        str_length = len(arrToCompare[i])
        if arrMax[i] < str_length:
            arrMax[i] = str_length
    return arrMax


def get_string(
    tableArray, maxArr
):  # gets string from array of arrays as a table (helper for __str__)
    temp_str = ""
    for row in tableArray:
        temp_str += "\n"
        for colNo in range(len(row)):
            max_col = maxArr[colNo]
            length = len(row[colNo])
            starting_loc = (max_col - length) // 2
            temp_str += (
                " " * starting_loc
                + row[colNo]
                + " " * (max_col - length - starting_loc)
            )
    return temp_str
