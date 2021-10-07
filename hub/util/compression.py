import re


def re_find_first(pattern, string):
    for match in re.finditer(pattern, string):
        return match
