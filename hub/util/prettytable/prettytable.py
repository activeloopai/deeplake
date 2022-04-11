#!/usr/bin/env python
#
# Copyright (c) 2009-2014, Luke Maurits <luke@maurits.id.au>
# All rights reserved.
# With contributions from:
#  * Chris Clark
#  * Klein Stephane
#  * John Filleau
#  * Vladimir Vrzić
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * The name of the author may not be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import copy
import csv
import io
import json
import math
import random
import re
import textwrap
from html import escape
from html.parser import HTMLParser

import wcwidth

# hrule styles
FRAME = 0
ALL = 1
NONE = 2
HEADER = 3

# Table styles
DEFAULT = 10
MSWORD_FRIENDLY = 11
PLAIN_COLUMNS = 12
MARKDOWN = 13
ORGMODE = 14
DOUBLE_BORDER = 15
SINGLE_BORDER = 16
RANDOM = 20

_re = re.compile(r"\033\[[0-9;]*m|\033\(B")


def _get_size(text):
    lines = text.split("\n")
    height = len(lines)
    width = max(_str_block_width(line) for line in lines)
    return width, height


class PrettyTable:
    def __init__(self, field_names=None, **kwargs):
        """Return a new PrettyTable instance

        Arguments:

        encoding - Unicode encoding scheme used to decode any encoded input
        title - optional table title
        field_names - list or tuple of field names
        fields - list or tuple of field names to include in displays
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        header - print a header showing field names (True or False)
        header_style - stylisation to apply to field names in header
            ("cap", "title", "upper", "lower" or None)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.
            Allowed values: FRAME, HEADER, ALL, NONE
        vrules - controls printing of vertical rules between columns.
            Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        custom_format - controls formatting of any column using callable
        min_table_width - minimum desired table width, in characters
        max_table_width - maximum desired table width, in characters
        min_width - minimum desired field width, in characters
        max_width - maximum desired field width, in characters
        padding_width - number of spaces on either side of column data
            (only used if left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        vertical_char - single character string used to draw vertical lines
        horizontal_char - single character string used to draw horizontal lines
        horizontal_align_char - single character string used to indicate alignment
        junction_char - single character string used to draw line junctions
        top_junction_char - single character string used to draw top line junctions
        bottom_junction_char -
            single character string used to draw bottom line junctions
        right_junction_char - single character string used to draw right line junctions
        left_junction_char - single character string used to draw left line junctions
        top_right_junction_char -
            single character string used to draw top-right line junctions
        top_left_junction_char -
            single character string used to draw top-left line junctions
        bottom_right_junction_char -
            single character string used to draw bottom-right line junctions
        bottom_left_junction_char -
            single character string used to draw bottom-left line junctions
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        align - default align for each column (None, "l", "c" or "r")
        valign - default valign for each row (None, "t", "m" or "b")
        reversesort - True or False to sort in descending or ascending order
        oldsortslice - Slice rows before sorting in the "old style" """

        self.encoding = kwargs.get("encoding", "UTF-8")

        # Data
        self._field_names = []
        self._rows = []
        self.align = {}
        self.valign = {}
        self.max_width = {}
        self.min_width = {}
        self.int_format = {}
        self.float_format = {}
        self.custom_format = {}

        if field_names:
            self.field_names = field_names
        else:
            self._widths = []

        # Options
        self._options = [
            "title",
            "start",
            "end",
            "fields",
            "header",
            "border",
            "sortby",
            "reversesort",
            "sort_key",
            "attributes",
            "format",
            "hrules",
            "vrules",
            "int_format",
            "float_format",
            "custom_format",
            "min_table_width",
            "max_table_width",
            "padding_width",
            "left_padding_width",
            "right_padding_width",
            "vertical_char",
            "horizontal_char",
            "horizontal_align_char",
            "junction_char",
            "header_style",
            "valign",
            "xhtml",
            "print_empty",
            "oldsortslice",
            "top_junction_char",
            "bottom_junction_char",
            "right_junction_char",
            "left_junction_char",
            "top_right_junction_char",
            "top_left_junction_char",
            "bottom_right_junction_char",
            "bottom_left_junction_char",
            "align",
            "valign",
            "max_width",
            "min_width",
            "none_format",
        ]
        for option in self._options:
            if option in kwargs:
                self._validate_option(option, kwargs[option])
            else:
                kwargs[option] = None

        self._title = kwargs["title"] or None
        self._start = kwargs["start"] or 0
        self._end = kwargs["end"] or None
        self._fields = kwargs["fields"] or None
        self._none_format = {}

        if kwargs["header"] in (True, False):
            self._header = kwargs["header"]
        else:
            self._header = True
        self._header_style = kwargs["header_style"] or None
        if kwargs["border"] in (True, False):
            self._border = kwargs["border"]
        else:
            self._border = True
        self._hrules = kwargs["hrules"] or FRAME
        self._vrules = kwargs["vrules"] or ALL

        self._sortby = kwargs["sortby"] or None
        if kwargs["reversesort"] in (True, False):
            self._reversesort = kwargs["reversesort"]
        else:
            self._reversesort = False
        self._sort_key = kwargs["sort_key"] or (lambda x: x)

        # Column specific arguments, use property.setters
        self.align = kwargs["align"] or {}
        self.valign = kwargs["valign"] or {}
        self.max_width = kwargs["max_width"] or {}
        self.min_width = kwargs["min_width"] or {}
        self.int_format = kwargs["int_format"] or {}
        self.float_format = kwargs["float_format"] or {}
        self.custom_format = kwargs["custom_format"] or {}
        self.none_format = kwargs["none_format"] or {}

        self._min_table_width = kwargs["min_table_width"] or None
        self._max_table_width = kwargs["max_table_width"] or None
        if kwargs["padding_width"] is None:
            self._padding_width = 1
        else:
            self._padding_width = kwargs["padding_width"]
        self._left_padding_width = kwargs["left_padding_width"] or None
        self._right_padding_width = kwargs["right_padding_width"] or None

        self._vertical_char = kwargs["vertical_char"] or "|"
        self._horizontal_char = kwargs["horizontal_char"] or "-"
        self._horizontal_align_char = kwargs["horizontal_align_char"]
        self._junction_char = kwargs["junction_char"] or "+"
        self._top_junction_char = kwargs["top_junction_char"]
        self._bottom_junction_char = kwargs["bottom_junction_char"]
        self._right_junction_char = kwargs["right_junction_char"]
        self._left_junction_char = kwargs["left_junction_char"]
        self._top_right_junction_char = kwargs["top_right_junction_char"]
        self._top_left_junction_char = kwargs["top_left_junction_char"]
        self._bottom_right_junction_char = kwargs["bottom_right_junction_char"]
        self._bottom_left_junction_char = kwargs["bottom_left_junction_char"]

        if kwargs["print_empty"] in (True, False):
            self._print_empty = kwargs["print_empty"]
        else:
            self._print_empty = True
        if kwargs["oldsortslice"] in (True, False):
            self._oldsortslice = kwargs["oldsortslice"]
        else:
            self._oldsortslice = False
        self._format = kwargs["format"] or False
        self._xhtml = kwargs["xhtml"] or False
        self._attributes = kwargs["attributes"] or {}

    def _justify(self, text, width, align):
        excess = width - _str_block_width(text)
        if align == "l":
            return text + excess * " "
        elif align == "r":
            return excess * " " + text
        else:
            if excess % 2:
                # Uneven padding
                # Put more space on right if text is of odd length...
                if _str_block_width(text) % 2:
                    return (excess // 2) * " " + text + (excess // 2 + 1) * " "
                # and more space on left if text is of even length
                else:
                    return (excess // 2 + 1) * " " + text + (excess // 2) * " "
                # Why distribute extra space this way?  To match the behaviour of
                # the inbuilt str.center() method.
            else:
                # Equal padding on either side
                return (excess // 2) * " " + text + (excess // 2) * " "

    def __getattr__(self, name):

        if name == "rowcount":
            return len(self._rows)
        elif name == "colcount":
            if self._field_names:
                return len(self._field_names)
            elif self._rows:
                return len(self._rows[0])
            else:
                return 0
        else:
            raise AttributeError(name)

    def __getitem__(self, index):

        new = PrettyTable()
        new.field_names = self.field_names
        for attr in self._options:
            setattr(new, "_" + attr, getattr(self, "_" + attr))
        setattr(new, "_align", getattr(self, "_align"))
        if isinstance(index, slice):
            for row in self._rows[index]:
                new.add_row(row)
        elif isinstance(index, int):
            new.add_row(self._rows[index])
        else:
            raise IndexError(f"Index {index} is invalid, must be an integer or slice")
        return new

    def __str__(self):
        return self.get_string()

    def __repr__(self):
        return self.get_string()

    def _repr_html_(self):
        """
        Returns get_html_string value by default
        as the repr call in Jupyter notebook environment
        """
        return self.get_html_string()

    ##############################
    # ATTRIBUTE VALIDATORS       #
    ##############################

    # The method _validate_option is all that should be used elsewhere in the code base
    # to validate options. It will call the appropriate validation method for that
    # option. The individual validation methods should never need to be called directly
    # (although nothing bad will happen if they *are*).
    # Validation happens in TWO places.
    # Firstly, in the property setters defined in the ATTRIBUTE MANAGEMENT section.
    # Secondly, in the _get_options method, where keyword arguments are mixed with
    # persistent settings

    def _validate_option(self, option, val):
        if option == "field_names":
            self._validate_field_names(val)
        elif option == "none_format":
            self._validate_none_format(val)
        elif option in (
            "start",
            "end",
            "max_width",
            "min_width",
            "min_table_width",
            "max_table_width",
            "padding_width",
            "left_padding_width",
            "right_padding_width",
            "format",
        ):
            self._validate_nonnegative_int(option, val)
        elif option == "sortby":
            self._validate_field_name(option, val)
        elif option == "sort_key":
            self._validate_function(option, val)
        elif option == "hrules":
            self._validate_hrules(option, val)
        elif option == "vrules":
            self._validate_vrules(option, val)
        elif option == "fields":
            self._validate_all_field_names(option, val)
        elif option in (
            "header",
            "border",
            "reversesort",
            "xhtml",
            "print_empty",
            "oldsortslice",
        ):
            self._validate_true_or_false(option, val)
        elif option == "header_style":
            self._validate_header_style(val)
        elif option == "int_format":
            self._validate_int_format(option, val)
        elif option == "float_format":
            self._validate_float_format(option, val)
        elif option == "custom_format":
            for k, formatter in val.items():
                self._validate_function(f"{option}.{k}", formatter)
        elif option in (
            "vertical_char",
            "horizontal_char",
            "horizontal_align_char",
            "junction_char",
            "top_junction_char",
            "bottom_junction_char",
            "right_junction_char",
            "left_junction_char",
            "top_right_junction_char",
            "top_left_junction_char",
            "bottom_right_junction_char",
            "bottom_left_junction_char",
        ):
            self._validate_single_char(option, val)
        elif option == "attributes":
            self._validate_attributes(option, val)

    def _validate_field_names(self, val):
        # Check for appropriate length
        if self._field_names:
            try:
                assert len(val) == len(self._field_names)
            except AssertionError:
                raise ValueError(
                    "Field name list has incorrect number of values, "
                    f"(actual) {len(val)}!={len(self._field_names)} (expected)"
                )
        if self._rows:
            try:
                assert len(val) == len(self._rows[0])
            except AssertionError:
                raise ValueError(
                    "Field name list has incorrect number of values, "
                    f"(actual) {len(val)}!={len(self._rows[0])} (expected)"
                )
        # Check for uniqueness
        try:
            assert len(val) == len(set(val))
        except AssertionError:
            raise ValueError("Field names must be unique")

    def _validate_none_format(self, val):
        try:
            if val is not None:
                assert isinstance(val, str)
        except AssertionError:
            raise TypeError(
                "Replacement for None value must be a string if being supplied."
            )

    def _validate_header_style(self, val):
        try:
            assert val in ("cap", "title", "upper", "lower", None)
        except AssertionError:
            raise ValueError(
                "Invalid header style, use cap, title, upper, lower or None"
            )

    def _validate_align(self, val):
        try:
            assert val in ["l", "c", "r"]
        except AssertionError:
            raise ValueError(f"Alignment {val} is invalid, use l, c or r")

    def _validate_valign(self, val):
        try:
            assert val in ["t", "m", "b", None]
        except AssertionError:
            raise ValueError(f"Alignment {val} is invalid, use t, m, b or None")

    def _validate_nonnegative_int(self, name, val):
        try:
            assert int(val) >= 0
        except AssertionError:
            raise ValueError(f"Invalid value for {name}: {val}")

    def _validate_true_or_false(self, name, val):
        try:
            assert val in (True, False)
        except AssertionError:
            raise ValueError(f"Invalid value for {name}. Must be True or False.")

    def _validate_int_format(self, name, val):
        if val == "":
            return
        try:
            assert isinstance(val, str)
            assert val.isdigit()
        except AssertionError:
            raise ValueError(
                f"Invalid value for {name}. Must be an integer format string."
            )

    def _validate_float_format(self, name, val):
        if val == "":
            return
        try:
            assert isinstance(val, str)
            assert "." in val
            bits = val.split(".")
            assert len(bits) <= 2
            assert bits[0] == "" or bits[0].isdigit()
            assert (
                bits[1] == ""
                or bits[1].isdigit()
                or (bits[1][-1] == "f" and bits[1].rstrip("f").isdigit())
            )
        except AssertionError:
            raise ValueError(
                f"Invalid value for {name}. Must be a float format string."
            )

    def _validate_function(self, name, val):
        try:
            assert hasattr(val, "__call__")
        except AssertionError:
            raise ValueError(f"Invalid value for {name}. Must be a function.")

    def _validate_hrules(self, name, val):
        try:
            assert val in (ALL, FRAME, HEADER, NONE)
        except AssertionError:
            raise ValueError(
                f"Invalid value for {name}. Must be ALL, FRAME, HEADER or NONE."
            )

    def _validate_vrules(self, name, val):
        try:
            assert val in (ALL, FRAME, NONE)
        except AssertionError:
            raise ValueError(f"Invalid value for {name}. Must be ALL, FRAME, or NONE.")

    def _validate_field_name(self, name, val):
        try:
            assert (val in self._field_names) or (val is None)
        except AssertionError:
            raise ValueError(f"Invalid field name: {val}")

    def _validate_all_field_names(self, name, val):
        try:
            for x in val:
                self._validate_field_name(name, x)
        except AssertionError:
            raise ValueError("Fields must be a sequence of field names")

    def _validate_single_char(self, name, val):
        try:
            assert _str_block_width(val) == 1
        except AssertionError:
            raise ValueError(f"Invalid value for {name}. Must be a string of length 1.")

    def _validate_attributes(self, name, val):
        try:
            assert isinstance(val, dict)
        except AssertionError:
            raise TypeError("Attributes must be a dictionary of name/value pairs")

    ##############################
    # ATTRIBUTE MANAGEMENT       #
    ##############################
    @property
    def rows(self):
        return self._rows[:]

    @property
    def xhtml(self):
        """Print <br/> tags if True, <br> tags if False"""
        return self._xhtml

    @xhtml.setter
    def xhtml(self, val):
        self._validate_option("xhtml", val)
        self._xhtml = val

    @property
    def none_format(self):
        return self._none_format

    @none_format.setter
    def none_format(self, val):
        if not self._field_names:
            self._none_format = {}
        elif val is None or (isinstance(val, dict) and len(val) == 0):
            for field in self._field_names:
                self._none_format[field] = None
        else:
            self._validate_none_format(val)
            for field in self._field_names:
                self._none_format[field] = val

    @property
    def field_names(self):
        """List or tuple of field names

        When setting field_names, if there are already field names the new list
        of field names must be the same length. Columns are renamed and row data
        remains unchanged."""
        return self._field_names

    @field_names.setter
    def field_names(self, val):
        val = [str(x) for x in val]
        self._validate_option("field_names", val)
        old_names = None
        if self._field_names:
            old_names = self._field_names[:]
        self._field_names = val
        if self._align and old_names:
            for old_name, new_name in zip(old_names, val):
                self._align[new_name] = self._align[old_name]
            for old_name in old_names:
                if old_name not in self._align:
                    self._align.pop(old_name)
        else:
            self.align = "c"
        if self._valign and old_names:
            for old_name, new_name in zip(old_names, val):
                self._valign[new_name] = self._valign[old_name]
            for old_name in old_names:
                if old_name not in self._valign:
                    self._valign.pop(old_name)
        else:
            self.valign = "t"

    @property
    def align(self):
        """Controls alignment of fields
        Arguments:

        align - alignment, one of "l", "c", or "r" """
        return self._align

    @align.setter
    def align(self, val):
        if not self._field_names:
            self._align = {}
        elif val is None or (isinstance(val, dict) and len(val) == 0):
            for field in self._field_names:
                self._align[field] = "c"
        else:
            self._validate_align(val)
            for field in self._field_names:
                self._align[field] = val

    @property
    def valign(self):
        """Controls vertical alignment of fields
        Arguments:

        valign - vertical alignment, one of "t", "m", or "b" """
        return self._valign

    @valign.setter
    def valign(self, val):
        if not self._field_names:
            self._valign = {}
        elif val is None or (isinstance(val, dict) and len(val) == 0):
            for field in self._field_names:
                self._valign[field] = "t"
        else:
            self._validate_valign(val)
            for field in self._field_names:
                self._valign[field] = val

    @property
    def max_width(self):
        """Controls maximum width of fields
        Arguments:

        max_width - maximum width integer"""
        return self._max_width

    @max_width.setter
    def max_width(self, val):
        if val is None or (isinstance(val, dict) and len(val) == 0):
            self._max_width = {}
        else:
            self._validate_option("max_width", val)
            for field in self._field_names:
                self._max_width[field] = val

    @property
    def min_width(self):
        """Controls minimum width of fields
        Arguments:

        min_width - minimum width integer"""
        return self._min_width

    @min_width.setter
    def min_width(self, val):
        if val is None or (isinstance(val, dict) and len(val) == 0):
            self._min_width = {}
        else:
            self._validate_option("min_width", val)
            for field in self._field_names:
                self._min_width[field] = val

    @property
    def min_table_width(self):
        return self._min_table_width

    @min_table_width.setter
    def min_table_width(self, val):
        self._validate_option("min_table_width", val)
        self._min_table_width = val

    @property
    def max_table_width(self):
        return self._max_table_width

    @max_table_width.setter
    def max_table_width(self, val):
        self._validate_option("max_table_width", val)
        self._max_table_width = val

    @property
    def fields(self):
        """List or tuple of field names to include in displays"""
        return self._fields

    @fields.setter
    def fields(self, val):
        self._validate_option("fields", val)
        self._fields = val

    @property
    def title(self):
        """Optional table title

        Arguments:

        title - table title"""
        return self._title

    @title.setter
    def title(self, val):
        self._title = str(val)

    @property
    def start(self):
        """Start index of the range of rows to print

        Arguments:

        start - index of first data row to include in output"""
        return self._start

    @start.setter
    def start(self, val):
        self._validate_option("start", val)
        self._start = val

    @property
    def end(self):
        """End index of the range of rows to print

        Arguments:

        end - index of last data row to include in output PLUS ONE (list slice style)"""
        return self._end

    @end.setter
    def end(self, val):
        self._validate_option("end", val)
        self._end = val

    @property
    def sortby(self):
        """Name of field by which to sort rows

        Arguments:

        sortby - field name to sort by"""
        return self._sortby

    @sortby.setter
    def sortby(self, val):
        self._validate_option("sortby", val)
        self._sortby = val

    @property
    def reversesort(self):
        """Controls direction of sorting (ascending vs descending)

        Arguments:

        reveresort - set to True to sort by descending order, or False to sort by
            ascending order"""
        return self._reversesort

    @reversesort.setter
    def reversesort(self, val):
        self._validate_option("reversesort", val)
        self._reversesort = val

    @property
    def sort_key(self):
        """Sorting key function, applied to data points before sorting

        Arguments:

        sort_key - a function which takes one argument and returns something to be
        sorted"""
        return self._sort_key

    @sort_key.setter
    def sort_key(self, val):
        self._validate_option("sort_key", val)
        self._sort_key = val

    @property
    def header(self):
        """Controls printing of table header with field names

        Arguments:

        header - print a header showing field names (True or False)"""
        return self._header

    @header.setter
    def header(self, val):
        self._validate_option("header", val)
        self._header = val

    @property
    def header_style(self):
        """Controls stylisation applied to field names in header

        Arguments:

        header_style - stylisation to apply to field names in header
            ("cap", "title", "upper", "lower" or None)"""
        return self._header_style

    @header_style.setter
    def header_style(self, val):
        self._validate_header_style(val)
        self._header_style = val

    @property
    def border(self):
        """Controls printing of border around table

        Arguments:

        border - print a border around the table (True or False)"""
        return self._border

    @border.setter
    def border(self, val):
        self._validate_option("border", val)
        self._border = val

    @property
    def hrules(self):
        """Controls printing of horizontal rules after rows

        Arguments:

        hrules - horizontal rules style.  Allowed values: FRAME, ALL, HEADER, NONE"""
        return self._hrules

    @hrules.setter
    def hrules(self, val):
        self._validate_option("hrules", val)
        self._hrules = val

    @property
    def vrules(self):
        """Controls printing of vertical rules between columns

        Arguments:

        vrules - vertical rules style.  Allowed values: FRAME, ALL, NONE"""
        return self._vrules

    @vrules.setter
    def vrules(self, val):
        self._validate_option("vrules", val)
        self._vrules = val

    @property
    def int_format(self):
        """Controls formatting of integer data
        Arguments:

        int_format - integer format string"""
        return self._int_format

    @int_format.setter
    def int_format(self, val):
        if val is None or (isinstance(val, dict) and len(val) == 0):
            self._int_format = {}
        else:
            self._validate_option("int_format", val)
            for field in self._field_names:
                self._int_format[field] = val

    @property
    def float_format(self):
        """Controls formatting of floating point data
        Arguments:

        float_format - floating point format string"""
        return self._float_format

    @float_format.setter
    def float_format(self, val):
        if val is None or (isinstance(val, dict) and len(val) == 0):
            self._float_format = {}
        else:
            self._validate_option("float_format", val)
            for field in self._field_names:
                self._float_format[field] = val

    @property
    def custom_format(self):
        """Controls formatting of any column using callable
        Arguments:

        custom_format - Dictionary of field_name and callable"""
        return self._custom_format

    @custom_format.setter
    def custom_format(self, val):
        if val is None:
            self._custom_format = {}
        elif isinstance(val, dict):
            for k, v in val.items():
                self._validate_function(f"custom_value.{k}", v)
            self._custom_format = val
        elif hasattr(val, "__call__"):
            self._validate_function("custom_value", val)
            for field in self._field_names:
                self._custom_format[field] = val
        else:
            raise TypeError(
                "The custom_format property need to be a dictionary or callable"
            )

    @property
    def padding_width(self):
        """The number of empty spaces between a column's edge and its content

        Arguments:

        padding_width - number of spaces, must be a positive integer"""
        return self._padding_width

    @padding_width.setter
    def padding_width(self, val):
        self._validate_option("padding_width", val)
        self._padding_width = val

    @property
    def left_padding_width(self):
        """The number of empty spaces between a column's left edge and its content

        Arguments:

        left_padding - number of spaces, must be a positive integer"""
        return self._left_padding_width

    @left_padding_width.setter
    def left_padding_width(self, val):
        self._validate_option("left_padding_width", val)
        self._left_padding_width = val

    @property
    def right_padding_width(self):
        """The number of empty spaces between a column's right edge and its content

        Arguments:

        right_padding - number of spaces, must be a positive integer"""
        return self._right_padding_width

    @right_padding_width.setter
    def right_padding_width(self, val):
        self._validate_option("right_padding_width", val)
        self._right_padding_width = val

    @property
    def vertical_char(self):
        """The character used when printing table borders to draw vertical lines

        Arguments:

        vertical_char - single character string used to draw vertical lines"""
        return self._vertical_char

    @vertical_char.setter
    def vertical_char(self, val):
        val = str(val)
        self._validate_option("vertical_char", val)
        self._vertical_char = val

    @property
    def horizontal_char(self):
        """The character used when printing table borders to draw horizontal lines

        Arguments:

        horizontal_char - single character string used to draw horizontal lines"""
        return self._horizontal_char

    @horizontal_char.setter
    def horizontal_char(self, val):
        val = str(val)
        self._validate_option("horizontal_char", val)
        self._horizontal_char = val

    @property
    def horizontal_align_char(self):
        """The character used to indicate column alignment in horizontal lines

        Arguments:

        horizontal_align_char - single character string used to indicate alignment"""
        return self._bottom_left_junction_char or self.junction_char

    @horizontal_align_char.setter
    def horizontal_align_char(self, val):
        val = str(val)
        self._validate_option("horizontal_align_char", val)
        self._horizontal_align_char = val

    @property
    def junction_char(self):
        """The character used when printing table borders to draw line junctions

        Arguments:

        junction_char - single character string used to draw line junctions"""
        return self._junction_char

    @junction_char.setter
    def junction_char(self, val):
        val = str(val)
        self._validate_option("junction_char", val)
        self._junction_char = val

    @property
    def top_junction_char(self):
        """The character used when printing table borders to draw top line junctions

        Arguments:

        top_junction_char - single character string used to draw top line junctions"""
        return self._top_junction_char or self.junction_char

    @top_junction_char.setter
    def top_junction_char(self, val):
        val = str(val)
        self._validate_option("top_junction_char", val)
        self._top_junction_char = val

    @property
    def bottom_junction_char(self):
        """The character used when printing table borders to draw bottom line junctions

        Arguments:

        bottom_junction_char -
            single character string used to draw bottom line junctions"""
        return self._bottom_junction_char or self.junction_char

    @bottom_junction_char.setter
    def bottom_junction_char(self, val):
        val = str(val)
        self._validate_option("bottom_junction_char", val)
        self._bottom_junction_char = val

    @property
    def right_junction_char(self):
        """The character used when printing table borders to draw right line junctions

        Arguments:

        right_junction_char -
            single character string used to draw right line junctions"""
        return self._right_junction_char or self.junction_char

    @right_junction_char.setter
    def right_junction_char(self, val):
        val = str(val)
        self._validate_option("right_junction_char", val)
        self._right_junction_char = val

    @property
    def left_junction_char(self):
        """The character used when printing table borders to draw left line junctions

        Arguments:

        left_junction_char - single character string used to draw left line junctions"""
        return self._left_junction_char or self.junction_char

    @left_junction_char.setter
    def left_junction_char(self, val):
        val = str(val)
        self._validate_option("left_junction_char", val)
        self._left_junction_char = val

    @property
    def top_right_junction_char(self):
        """The character used when printing table borders to draw top-right line junctions

        Arguments:

        top_right_junction_char -
            single character string used to draw top-right line junctions"""
        return self._top_right_junction_char or self.junction_char

    @top_right_junction_char.setter
    def top_right_junction_char(self, val):
        val = str(val)
        self._validate_option("top_right_junction_char", val)
        self._top_right_junction_char = val

    @property
    def top_left_junction_char(self):
        """
        The character used when printing table borders to draw top-left line junctions

        Arguments:

        top_left_junction_char -
            single character string used to draw top-left line junctions"""
        return self._top_left_junction_char or self.junction_char

    @top_left_junction_char.setter
    def top_left_junction_char(self, val):
        val = str(val)
        self._validate_option("top_left_junction_char", val)
        self._top_left_junction_char = val

    @property
    def bottom_right_junction_char(self):
        """The character used when printing table borders
           to draw bottom-right line junctions

        Arguments:

        bottom_right_junction_char -
            single character string used to draw bottom-right line junctions"""
        return self._bottom_right_junction_char or self.junction_char

    @bottom_right_junction_char.setter
    def bottom_right_junction_char(self, val):
        val = str(val)
        self._validate_option("bottom_right_junction_char", val)
        self._bottom_right_junction_char = val

    @property
    def bottom_left_junction_char(self):
        """The character used when printing table borders
           to draw bottom-left line junctions

        Arguments:

        bottom_left_junction_char -
            single character string used to draw bottom-left line junctions"""
        return self._bottom_left_junction_char or self.junction_char

    @bottom_left_junction_char.setter
    def bottom_left_junction_char(self, val):
        val = str(val)
        self._validate_option("bottom_left_junction_char", val)
        self._bottom_left_junction_char = val

    @property
    def format(self):
        """Controls whether or not HTML tables are formatted to match styling options

        Arguments:

        format - True or False"""
        return self._format

    @format.setter
    def format(self, val):
        self._validate_option("format", val)
        self._format = val

    @property
    def print_empty(self):
        """Controls whether or not empty tables produce a header and frame or just an
        empty string

        Arguments:

        print_empty - True or False"""
        return self._print_empty

    @print_empty.setter
    def print_empty(self, val):
        self._validate_option("print_empty", val)
        self._print_empty = val

    @property
    def attributes(self):
        """A dictionary of HTML attribute name/value pairs to be included in the
        <table> tag when printing HTML

        Arguments:

        attributes - dictionary of attributes"""
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        self._validate_option("attributes", val)
        self._attributes = val

    @property
    def oldsortslice(self):
        """oldsortslice - Slice rows before sorting in the "old style" """
        return self._oldsortslice

    @oldsortslice.setter
    def oldsortslice(self, val):
        self._validate_option("oldsortslice", val)
        self._oldsortslice = val

    ##############################
    # OPTION MIXER               #
    ##############################

    def _get_options(self, kwargs):

        options = {}
        for option in self._options:
            if option in kwargs:
                self._validate_option(option, kwargs[option])
                options[option] = kwargs[option]
            else:
                options[option] = getattr(self, option)
        return options

    ##############################
    # PRESET STYLE LOGIC         #
    ##############################

    def set_style(self, style):

        if style == DEFAULT:
            self._set_default_style()
        elif style == MSWORD_FRIENDLY:
            self._set_msword_style()
        elif style == PLAIN_COLUMNS:
            self._set_columns_style()
        elif style == MARKDOWN:
            self._set_markdown_style()
        elif style == ORGMODE:
            self._set_orgmode_style()
        elif style == DOUBLE_BORDER:
            self._set_double_border_style()
        elif style == SINGLE_BORDER:
            self._set_single_border_style()
        elif style == RANDOM:
            self._set_random_style()
        else:
            raise ValueError("Invalid pre-set style")

    def _set_orgmode_style(self):
        self._set_default_style()
        self.orgmode = True

    def _set_markdown_style(self):
        self.header = True
        self.border = True
        self._hrules = None
        self.padding_width = 1
        self.left_padding_width = 1
        self.right_padding_width = 1
        self.vertical_char = "|"
        self.junction_char = "|"
        self._horizontal_align_char = ":"

    def _set_default_style(self):

        self.header = True
        self.border = True
        self._hrules = FRAME
        self._vrules = ALL
        self.padding_width = 1
        self.left_padding_width = 1
        self.right_padding_width = 1
        self.vertical_char = "|"
        self.horizontal_char = "-"
        self._horizontal_align_char = None
        self.junction_char = "+"
        self._top_junction_char = None
        self._bottom_junction_char = None
        self._right_junction_char = None
        self._left_junction_char = None
        self._top_right_junction_char = None
        self._top_left_junction_char = None
        self._bottom_right_junction_char = None
        self._bottom_left_junction_char = None

    def _set_msword_style(self):

        self.header = True
        self.border = True
        self._hrules = NONE
        self.padding_width = 1
        self.left_padding_width = 1
        self.right_padding_width = 1
        self.vertical_char = "|"

    def _set_columns_style(self):

        self.header = True
        self.border = False
        self.padding_width = 1
        self.left_padding_width = 0
        self.right_padding_width = 8

    def _set_double_border_style(self):
        self.horizontal_char = "═"
        self.vertical_char = "║"
        self.junction_char = "╬"
        self.top_junction_char = "╦"
        self.bottom_junction_char = "╩"
        self.right_junction_char = "╣"
        self.left_junction_char = "╠"
        self.top_right_junction_char = "╗"
        self.top_left_junction_char = "╔"
        self.bottom_right_junction_char = "╝"
        self.bottom_left_junction_char = "╚"

    def _set_single_border_style(self):
        self.horizontal_char = "─"
        self.vertical_char = "│"
        self.junction_char = "┼"
        self.top_junction_char = "┬"
        self.bottom_junction_char = "┴"
        self.right_junction_char = "┤"
        self.left_junction_char = "├"
        self.top_right_junction_char = "┐"
        self.top_left_junction_char = "┌"
        self.bottom_right_junction_char = "┘"
        self.bottom_left_junction_char = "└"

    def _set_random_style(self):

        # Just for fun!
        self.header = random.choice((True, False))
        self.border = random.choice((True, False))
        self._hrules = random.choice((ALL, FRAME, HEADER, NONE))
        self._vrules = random.choice((ALL, FRAME, NONE))
        self.left_padding_width = random.randint(0, 5)
        self.right_padding_width = random.randint(0, 5)
        self.vertical_char = random.choice(r"~!@#$%^&*()_+|-=\{}[];':\",./;<>?")
        self.horizontal_char = random.choice(r"~!@#$%^&*()_+|-=\{}[];':\",./;<>?")
        self.junction_char = random.choice(r"~!@#$%^&*()_+|-=\{}[];':\",./;<>?")

    ##############################
    # DATA INPUT METHODS         #
    ##############################

    def add_rows(self, rows):

        """Add rows to the table

        Arguments:

        rows - rows of data, should be an iterable of lists, each list with as many
        elements as the table has fields"""
        for row in rows:
            self.add_row(row)

    def add_row(self, row):

        """Add a row to the table

        Arguments:

        row - row of data, should be a list with as many elements as the table
        has fields"""

        if self._field_names and len(row) != len(self._field_names):
            raise ValueError(
                "Row has incorrect number of values, "
                f"(actual) {len(row)}!={len(self._field_names)} (expected)"
            )
        if not self._field_names:
            self.field_names = [f"Field {n + 1}" for n in range(0, len(row))]
        self._rows.append(list(row))

    def del_row(self, row_index):

        """Delete a row from the table

        Arguments:

        row_index - The index of the row you want to delete.  Indexing starts at 0."""

        if row_index > len(self._rows) - 1:
            raise IndexError(
                f"Can't delete row at index {row_index}, "
                f"table only has {len(self._rows)} rows"
            )
        del self._rows[row_index]

    def add_column(self, fieldname, column, align="c", valign="t"):

        """Add a column to the table.

        Arguments:

        fieldname - name of the field to contain the new column of data
        column - column of data, should be a list with as many elements as the
        table has rows
        align - desired alignment for this column - "l" for left, "c" for centre and
            "r" for right
        valign - desired vertical alignment for new columns - "t" for top,
            "m" for middle and "b" for bottom"""

        if len(self._rows) in (0, len(column)):
            self._validate_align(align)
            self._validate_valign(valign)
            self._field_names.append(fieldname)
            self._align[fieldname] = align
            self._valign[fieldname] = valign
            for i in range(0, len(column)):
                if len(self._rows) < i + 1:
                    self._rows.append([])
                self._rows[i].append(column[i])
        else:
            raise ValueError(
                f"Column length {len(column)} does not match number of rows "
                f"{len(self._rows)}"
            )

    def add_autoindex(self, fieldname="Index"):
        """Add an auto-incrementing index column to the table.
        Arguments:
        fieldname - name of the field to contain the new column of data"""
        self._field_names.insert(0, fieldname)
        self._align[fieldname] = self.align
        self._valign[fieldname] = self.valign
        for i, row in enumerate(self._rows):
            row.insert(0, i + 1)

    def del_column(self, fieldname):

        """Delete a column from the table

        Arguments:

        fieldname - The field name of the column you want to delete."""

        if fieldname not in self._field_names:
            raise ValueError(
                "Can't delete column %r which is not a field name of this table."
                " Field names are: %s"
                % (fieldname, ", ".join(map(repr, self._field_names)))
            )

        col_index = self._field_names.index(fieldname)
        del self._field_names[col_index]
        for row in self._rows:
            del row[col_index]

    def clear_rows(self):

        """Delete all rows from the table but keep the current field names"""

        self._rows = []

    def clear(self):

        """Delete all rows and field names from the table, maintaining nothing but
        styling options"""

        self._rows = []
        self._field_names = []
        self._widths = []

    ##############################
    # MISC PUBLIC METHODS        #
    ##############################

    def copy(self):
        return copy.deepcopy(self)

    ##############################
    # MISC PRIVATE METHODS       #
    ##############################

    def _format_value(self, field, value):
        if isinstance(value, int) and field in self._int_format:
            return ("%%%sd" % self._int_format[field]) % value
        elif isinstance(value, float) and field in self._float_format:
            return ("%%%sf" % self._float_format[field]) % value

        formatter = self._custom_format.get(field, (lambda f, v: str(v)))
        return formatter(field, value)

    def _compute_table_width(self, options):
        table_width = 2 if options["vrules"] in (FRAME, ALL) else 0
        per_col_padding = sum(self._get_padding_widths(options))
        for index, fieldname in enumerate(self.field_names):
            if not options["fields"] or (
                options["fields"] and fieldname in options["fields"]
            ):
                table_width += self._widths[index] + per_col_padding
        return table_width

    def _compute_widths(self, rows, options):
        if options["header"]:
            widths = [_get_size(field)[0] for field in self._field_names]
        else:
            widths = len(self.field_names) * [0]

        for row in rows:
            for index, value in enumerate(row):
                fieldname = self.field_names[index]
                if self.none_format.get(fieldname) is not None:
                    if value == "None" or value is None:
                        value = self.none_format.get(fieldname)
                if fieldname in self.max_width:
                    widths[index] = max(
                        widths[index],
                        min(_get_size(value)[0], self.max_width[fieldname]),
                    )
                else:
                    widths[index] = max(widths[index], _get_size(value)[0])
                if fieldname in self.min_width:
                    widths[index] = max(widths[index], self.min_width[fieldname])
        self._widths = widths

        # Are we exceeding max_table_width?
        if self._max_table_width:
            table_width = self._compute_table_width(options)
            if table_width > self._max_table_width:
                # Shrink widths in proportion
                scale = 1.0 * self._max_table_width / table_width
                widths = [int(math.floor(w * scale)) for w in widths]
                self._widths = widths

        # Are we under min_table_width or title width?
        if self._min_table_width or options["title"]:
            if options["title"]:
                title_width = len(options["title"]) + sum(
                    self._get_padding_widths(options)
                )
                if options["vrules"] in (FRAME, ALL):
                    title_width += 2
            else:
                title_width = 0
            min_table_width = self.min_table_width or 0
            min_width = max(title_width, min_table_width)
            table_width = self._compute_table_width(options)
            if table_width < min_width:
                # Grow widths in proportion
                scale = 1.0 * min_width / table_width
                widths = [int(math.ceil(w * scale)) for w in widths]
                self._widths = widths

    def _get_padding_widths(self, options):

        if options["left_padding_width"] is not None:
            lpad = options["left_padding_width"]
        else:
            lpad = options["padding_width"]
        if options["right_padding_width"] is not None:
            rpad = options["right_padding_width"]
        else:
            rpad = options["padding_width"]
        return lpad, rpad

    def _get_rows(self, options):
        """Return only those data rows that should be printed, based on slicing and
        sorting.

        Arguments:

        options - dictionary of option settings."""

        if options["oldsortslice"]:
            rows = copy.deepcopy(self._rows[options["start"] : options["end"]])
        else:
            rows = copy.deepcopy(self._rows)

        # Sort
        if options["sortby"]:
            sortindex = self._field_names.index(options["sortby"])
            # Decorate
            rows = [[row[sortindex]] + row for row in rows]
            # Sort
            rows.sort(reverse=options["reversesort"], key=options["sort_key"])
            # Undecorate
            rows = [row[1:] for row in rows]

        # Slice if necessary
        if not options["oldsortslice"]:
            rows = rows[options["start"] : options["end"]]

        return rows

    def _format_row(self, row):
        return [
            self._format_value(field, value)
            for (field, value) in zip(self._field_names, row)
        ]

    def _format_rows(self, rows):
        return [self._format_row(row) for row in rows]

    ##############################
    # PLAIN TEXT STRING METHODS  #
    ##############################

    def get_string(self, **kwargs):

        """Return string representation of table in current state.

        Arguments:

        title - optional table title
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.
            Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.
            Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        custom_format - controls formatting of any column using callable
        padding_width - number of spaces on either side of column data (only used if
            left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        vertical_char - single character string used to draw vertical lines
        horizontal_char - single character string used to draw horizontal lines
        horizontal_align_char - single character string used to indicate alignment
        junction_char - single character string used to draw line junctions
        junction_char - single character string used to draw line junctions
        top_junction_char - single character string used to draw top line junctions
        bottom_junction_char -
            single character string used to draw bottom line junctions
        right_junction_char - single character string used to draw right line junctions
        left_junction_char - single character string used to draw left line junctions
        top_right_junction_char -
            single character string used to draw top-right line junctions
        top_left_junction_char -
            single character string used to draw top-left line junctions
        bottom_right_junction_char -
            single character string used to draw bottom-right line junctions
        bottom_left_junction_char -
            single character string used to draw bottom-left line junctions
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        reversesort - True or False to sort in descending or ascending order
        print empty - if True, stringify just the header for an empty table,
            if False return an empty string"""

        options = self._get_options(kwargs)

        lines = []

        # Don't think too hard about an empty table
        # Is this the desired behaviour?  Maybe we should still print the header?
        if self.rowcount == 0 and (not options["print_empty"] or not options["border"]):
            return ""

        # Get the rows we need to print, taking into account slicing, sorting, etc.
        rows = self._get_rows(options)

        # Turn all data in all rows into Unicode, formatted as desired
        formatted_rows = self._format_rows(rows)

        # Compute column widths
        self._compute_widths(formatted_rows, options)
        self._hrule = self._stringify_hrule(options)

        # Add title
        title = options["title"] or self._title
        if title:
            lines.append(self._stringify_title(title, options))

        # Add header or top of border
        if options["header"]:
            lines.append(self._stringify_header(options))
        elif options["border"] and options["hrules"] in (ALL, FRAME):
            lines.append(self._stringify_hrule(options, where="top_"))
            if title and options["vrules"] in (ALL, FRAME):
                lines[-1] = (
                    self.left_junction_char + lines[-1][1:-1] + self.right_junction_char
                )

        # Add rows
        for row in formatted_rows[:-1]:
            lines.append(self._stringify_row(row, options, self._hrule))
        if formatted_rows:
            lines.append(
                self._stringify_row(
                    formatted_rows[-1],
                    options,
                    self._stringify_hrule(options, where="bottom_"),
                )
            )

        # Add bottom of border
        if options["border"] and options["hrules"] == FRAME:
            lines.append(self._stringify_hrule(options, where="bottom_"))

        if "orgmode" in self.__dict__ and self.orgmode is True:
            tmp = list()
            for line in lines:
                tmp.extend(line.split("\n"))
            lines = ["|" + line[1:-1] + "|" for line in tmp]

        return "\n".join(lines)

    def _stringify_hrule(self, options, where=""):

        if not options["border"]:
            return ""
        lpad, rpad = self._get_padding_widths(options)
        if options["vrules"] in (ALL, FRAME):
            bits = [options[where + "left_junction_char"]]
        else:
            bits = [options["horizontal_char"]]
        # For tables with no data or fieldnames
        if not self._field_names:
            bits.append(options[where + "right_junction_char"])
            return "".join(bits)
        for field, width in zip(self._field_names, self._widths):
            if options["fields"] and field not in options["fields"]:
                continue

            line = (width + lpad + rpad) * options["horizontal_char"]

            # If necessary, add column alignment characters (e.g. ":" for Markdown)
            if self._horizontal_align_char:
                if self._align[field] in ("l", "c"):
                    line = self._horizontal_align_char + line[1:]
                if self._align[field] in ("c", "r"):
                    line = line[:-1] + self._horizontal_align_char

            bits.append(line)
            if options["vrules"] == ALL:
                bits.append(options[where + "junction_char"])
            else:
                bits.append(options["horizontal_char"])
        if options["vrules"] in (ALL, FRAME):
            bits.pop()
            bits.append(options[where + "right_junction_char"])
        return "".join(bits)

    def _stringify_title(self, title, options):

        lines = []
        lpad, rpad = self._get_padding_widths(options)
        if options["border"]:
            if options["vrules"] == ALL:
                options["vrules"] = FRAME
                lines.append(self._stringify_hrule(options, "top_"))
                options["vrules"] = ALL
            elif options["vrules"] == FRAME:
                lines.append(self._stringify_hrule(options, "top_"))
        bits = []
        endpoint = (
            options["vertical_char"] if options["vrules"] in (ALL, FRAME) else " "
        )
        bits.append(endpoint)
        title = " " * lpad + title + " " * rpad
        bits.append(self._justify(title, len(self._hrule) - 2, "c"))
        bits.append(endpoint)
        lines.append("".join(bits))
        return "\n".join(lines)

    def _stringify_header(self, options):

        bits = []
        lpad, rpad = self._get_padding_widths(options)
        if options["border"]:
            if options["hrules"] in (ALL, FRAME):
                bits.append(self._stringify_hrule(options, "top_"))
                if options["title"] and options["vrules"] in (ALL, FRAME):
                    bits[-1] = (
                        self.left_junction_char
                        + bits[-1][1:-1]
                        + self.right_junction_char
                    )
                bits.append("\n")
            if options["vrules"] in (ALL, FRAME):
                bits.append(options["vertical_char"])
            else:
                bits.append(" ")
        # For tables with no data or field names
        if not self._field_names:
            if options["vrules"] in (ALL, FRAME):
                bits.append(options["vertical_char"])
            else:
                bits.append(" ")
        for (field, width) in zip(self._field_names, self._widths):
            if options["fields"] and field not in options["fields"]:
                continue
            if self._header_style == "cap":
                fieldname = field.capitalize()
            elif self._header_style == "title":
                fieldname = field.title()
            elif self._header_style == "upper":
                fieldname = field.upper()
            elif self._header_style == "lower":
                fieldname = field.lower()
            else:
                fieldname = field
            if _str_block_width(fieldname) > width:
                fieldname = fieldname[:width]
            bits.append(
                " " * lpad
                + self._justify(fieldname, width, self._align[field])
                + " " * rpad
            )
            if options["border"]:
                if options["vrules"] == ALL:
                    bits.append(options["vertical_char"])
                else:
                    bits.append(" ")
        # If vrules is FRAME, then we just appended a space at the end
        # of the last field, when we really want a vertical character
        if options["border"] and options["vrules"] == FRAME:
            bits.pop()
            bits.append(options["vertical_char"])
        if options["border"] and options["hrules"] != NONE:
            bits.append("\n")
            bits.append(self._hrule)
        return "".join(bits)

    def _stringify_row(self, row, options, hrule):

        for (index, field, value, width) in zip(
            range(0, len(row)), self._field_names, row, self._widths
        ):
            # Enforce max widths
            lines = value.split("\n")
            new_lines = []
            for line in lines:
                if line == "None" and self.none_format.get(field) is not None:
                    line = self.none_format[field]
                if _str_block_width(line) > width:
                    line = textwrap.fill(line, width)
                new_lines.append(line)
            lines = new_lines
            value = "\n".join(lines)
            row[index] = value

        row_height = 0
        for c in row:
            h = _get_size(c)[1]
            if h > row_height:
                row_height = h

        bits = []
        lpad, rpad = self._get_padding_widths(options)
        for y in range(0, row_height):
            bits.append([])
            if options["border"]:
                if options["vrules"] in (ALL, FRAME):
                    bits[y].append(self.vertical_char)
                else:
                    bits[y].append(" ")

        for (field, value, width) in zip(self._field_names, row, self._widths):

            valign = self._valign[field]
            lines = value.split("\n")
            d_height = row_height - len(lines)
            if d_height:
                if valign == "m":
                    lines = (
                        [""] * int(d_height / 2)
                        + lines
                        + [""] * (d_height - int(d_height / 2))
                    )
                elif valign == "b":
                    lines = [""] * d_height + lines
                else:
                    lines = lines + [""] * d_height

            y = 0
            for line in lines:
                if options["fields"] and field not in options["fields"]:
                    continue

                bits[y].append(
                    " " * lpad
                    + self._justify(line, width, self._align[field])
                    + " " * rpad
                )
                if options["border"]:
                    if options["vrules"] == ALL:
                        bits[y].append(self.vertical_char)
                    else:
                        bits[y].append(" ")
                y += 1

        # If vrules is FRAME, then we just appended a space at the end
        # of the last field, when we really want a vertical character
        for y in range(0, row_height):
            if options["border"] and options["vrules"] == FRAME:
                bits[y].pop()
                bits[y].append(options["vertical_char"])

        if options["border"] and options["hrules"] == ALL:
            bits[row_height - 1].append("\n")
            bits[row_height - 1].append(hrule)

        for y in range(0, row_height):
            bits[y] = "".join(bits[y])

        return "\n".join(bits)

    def paginate(self, page_length=58, **kwargs):

        pages = []
        kwargs["start"] = kwargs.get("start", 0)
        true_end = kwargs.get("end", self.rowcount)
        while True:
            kwargs["end"] = min(kwargs["start"] + page_length, true_end)
            pages.append(self.get_string(**kwargs))
            if kwargs["end"] == true_end:
                break
            kwargs["start"] += page_length
        return "\f".join(pages)

    ##############################
    # CSV STRING METHODS         #
    ##############################
    def get_csv_string(self, **kwargs):

        """Return string representation of CSV formatted table in the current state

        Keyword arguments are first interpreted as table formatting options, and
        then any unused keyword arguments are passed to csv.writer(). For
        example, get_csv_string(header=False, delimiter='\t') would use
        header as a PrettyTable formatting option (skip the header row) and
        delimiter as a csv.writer keyword argument.
        """

        options = self._get_options(kwargs)
        csv_options = {
            key: value for key, value in kwargs.items() if key not in options
        }
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer, **csv_options)

        if options.get("header"):
            csv_writer.writerow(self._field_names)
        for row in self._get_rows(options):
            csv_writer.writerow(row)

        return csv_buffer.getvalue()

    ##############################
    # JSON STRING METHODS        #
    ##############################
    def get_json_string(self, **kwargs):

        """Return string representation of JSON formatted table in the current state

        Keyword arguments are first interpreted as table formatting options, and
        then any unused keyword arguments are passed to json.dumps(). For
        example, get_json_string(header=False, indent=2) would use header as
        a PrettyTable formatting option (skip the header row) and indent as a
        json.dumps keyword argument.
        """

        options = self._get_options(kwargs)
        json_options = dict(indent=4, separators=(",", ": "), sort_keys=True)
        json_options.update(
            {key: value for key, value in kwargs.items() if key not in options}
        )
        objects = []

        if options.get("header"):
            objects.append(self.field_names)
        for row in self._get_rows(options):
            objects.append(dict(zip(self._field_names, row)))

        return json.dumps(objects, **json_options)

    ##############################
    # HTML STRING METHODS        #
    ##############################

    def get_html_string(self, **kwargs):
        """Return string representation of HTML formatted version of table in current
        state.

        Arguments:

        title - optional table title
        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.
            Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.
            Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        custom_format - controls formatting of any column using callable
        padding_width - number of spaces on either side of column data (only used if
            left and right paddings are None)
        left_padding_width - number of spaces on left hand side of column data
        right_padding_width - number of spaces on right hand side of column data
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        attributes - dictionary of name/value pairs to include as HTML attributes in the
            <table> tag
        format - Controls whether or not HTML tables are formatted to match
            styling options (True or False)
        xhtml - print <br/> tags if True, <br> tags if False"""

        options = self._get_options(kwargs)

        if options["format"]:
            string = self._get_formatted_html_string(options)
        else:
            string = self._get_simple_html_string(options)

        return string

    def _get_simple_html_string(self, options):

        lines = []
        if options["xhtml"]:
            linebreak = "<br/>"
        else:
            linebreak = "<br>"

        open_tag = ["<table"]
        if options["attributes"]:
            for attr_name in options["attributes"]:
                open_tag.append(f' {attr_name}="{options["attributes"][attr_name]}"')
        open_tag.append(">")
        lines.append("".join(open_tag))

        # Title
        title = options["title"] or self._title
        if title:
            lines.append(f"    <caption>{title}</caption>")

        # Headers
        if options["header"]:
            lines.append("    <thead>")
            lines.append("        <tr>")
            for field in self._field_names:
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append(
                    "            <th>%s</th>" % escape(field).replace("\n", linebreak)
                )
            lines.append("        </tr>")
            lines.append("    </thead>")

        # Data
        lines.append("    <tbody>")
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows)
        for row in formatted_rows:
            lines.append("        <tr>")
            for field, datum in zip(self._field_names, row):
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append(
                    "            <td>%s</td>" % escape(datum).replace("\n", linebreak)
                )
            lines.append("        </tr>")
        lines.append("    </tbody>")
        lines.append("</table>")

        return "\n".join(lines)

    def _get_formatted_html_string(self, options):

        lines = []
        lpad, rpad = self._get_padding_widths(options)
        if options["xhtml"]:
            linebreak = "<br/>"
        else:
            linebreak = "<br>"

        open_tag = ["<table"]
        if options["border"]:
            if options["hrules"] == ALL and options["vrules"] == ALL:
                open_tag.append(' frame="box" rules="all"')
            elif options["hrules"] == FRAME and options["vrules"] == FRAME:
                open_tag.append(' frame="box"')
            elif options["hrules"] == FRAME and options["vrules"] == ALL:
                open_tag.append(' frame="box" rules="cols"')
            elif options["hrules"] == FRAME:
                open_tag.append(' frame="hsides"')
            elif options["hrules"] == ALL:
                open_tag.append(' frame="hsides" rules="rows"')
            elif options["vrules"] == FRAME:
                open_tag.append(' frame="vsides"')
            elif options["vrules"] == ALL:
                open_tag.append(' frame="vsides" rules="cols"')
        if options["attributes"]:
            for attr_name in options["attributes"]:
                open_tag.append(f' {attr_name}="{options["attributes"][attr_name]}"')
        open_tag.append(">")
        lines.append("".join(open_tag))

        # Title
        title = options["title"] or self._title
        if title:
            lines.append(f"    <caption>{title}</caption>")

        # Headers
        if options["header"]:
            lines.append("    <thead>")
            lines.append("        <tr>")
            for field in self._field_names:
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append(
                    '            <th style="padding-left: %dem; padding-right: %dem; text-align: center">%s</th>'  # noqa: E501
                    % (lpad, rpad, escape(field).replace("\n", linebreak))
                )
            lines.append("        </tr>")
            lines.append("    </thead>")

        # Data
        lines.append("    <tbody>")
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows)
        aligns = []
        valigns = []
        for field in self._field_names:
            aligns.append(
                {"l": "left", "r": "right", "c": "center"}[self._align[field]]
            )
            valigns.append(
                {"t": "top", "m": "middle", "b": "bottom"}[self._valign[field]]
            )
        for row in formatted_rows:
            lines.append("        <tr>")
            for field, datum, align, valign in zip(
                self._field_names, row, aligns, valigns
            ):
                if options["fields"] and field not in options["fields"]:
                    continue
                lines.append(
                    '            <td style="padding-left: %dem; padding-right: %dem; text-align: %s; vertical-align: %s">%s</td>'  # noqa: E501
                    % (
                        lpad,
                        rpad,
                        align,
                        valign,
                        escape(datum).replace("\n", linebreak),
                    )
                )
            lines.append("        </tr>")
        lines.append("    </tbody>")
        lines.append("</table>")

        return "\n".join(lines)

    ##############################
    # LATEX STRING METHODS       #
    ##############################

    def get_latex_string(self, **kwargs):
        """Return string representation of LaTex formatted version of table in current
        state.

        Arguments:

        start - index of first data row to include in output
        end - index of last data row to include in output PLUS ONE (list slice style)
        fields - names of fields (columns) to include
        header - print a header showing field names (True or False)
        border - print a border around the table (True or False)
        hrules - controls printing of horizontal rules after rows.
            Allowed values: ALL, FRAME, HEADER, NONE
        vrules - controls printing of vertical rules between columns.
            Allowed values: FRAME, ALL, NONE
        int_format - controls formatting of integer data
        float_format - controls formatting of floating point data
        sortby - name of field to sort rows by
        sort_key - sorting key function, applied to data points before sorting
        format - Controls whether or not HTML tables are formatted to match
            styling options (True or False)
        """
        options = self._get_options(kwargs)

        if options["format"]:
            string = self._get_formatted_latex_string(options)
        else:
            string = self._get_simple_latex_string(options)
        return string

    def _get_simple_latex_string(self, options):
        lines = []

        wanted_fields = []
        if options["fields"]:
            wanted_fields = [
                field for field in self._field_names if field in options["fields"]
            ]
        else:
            wanted_fields = self._field_names

        alignments = "".join([self._align[field] for field in wanted_fields])

        begin_cmd = "\\begin{tabular}{%s}" % alignments
        lines.append(begin_cmd)

        # Headers
        if options["header"]:
            lines.append(" & ".join(wanted_fields) + " \\\\")

        # Data
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows)
        for row in formatted_rows:
            wanted_data = [
                d for f, d in zip(self._field_names, row) if f in wanted_fields
            ]
            lines.append(" & ".join(wanted_data) + " \\\\")

        lines.append("\\end{tabular}")

        return "\r\n".join(lines)

    def _get_formatted_latex_string(self, options):
        lines = []

        wanted_fields = []
        if options["fields"]:
            wanted_fields = [
                field for field in self._field_names if field in options["fields"]
            ]
        else:
            wanted_fields = self._field_names

        wanted_alignments = [self._align[field] for field in wanted_fields]
        if options["border"] and options["vrules"] == ALL:
            alignment_str = "|".join(wanted_alignments)
        else:
            alignment_str = "".join(wanted_alignments)

        if options["border"] and options["vrules"] in [ALL, FRAME]:
            alignment_str = "|" + alignment_str + "|"

        begin_cmd = "\\begin{tabular}{%s}" % alignment_str
        lines.append(begin_cmd)

        if options["border"] and options["hrules"] in [ALL, FRAME]:
            lines.append("\\hline")

        # Headers
        if options["header"]:
            lines.append(" & ".join(wanted_fields) + " \\\\")
        if options["border"] and options["hrules"] in [ALL, HEADER]:
            lines.append("\\hline")

        # Data
        rows = self._get_rows(options)
        formatted_rows = self._format_rows(rows)
        rows = self._get_rows(options)
        for row in formatted_rows:
            wanted_data = [
                d for f, d in zip(self._field_names, row) if f in wanted_fields
            ]
            lines.append(" & ".join(wanted_data) + " \\\\")
            if options["border"] and options["hrules"] == ALL:
                lines.append("\\hline")

        if options["border"] and options["hrules"] == FRAME:
            lines.append("\\hline")

        lines.append("\\end{tabular}")

        return "\r\n".join(lines)


##############################
# UNICODE WIDTH FUNCTION     #
##############################


def _str_block_width(val):
    return wcwidth.wcswidth(_re.sub("", val))


##############################
# TABLE FACTORIES            #
##############################


def from_csv(fp, field_names=None, **kwargs):
    fmtparams = {}
    for param in [
        "delimiter",
        "doublequote",
        "escapechar",
        "lineterminator",
        "quotechar",
        "quoting",
        "skipinitialspace",
        "strict",
    ]:
        if param in kwargs:
            fmtparams[param] = kwargs.pop(param)
    if fmtparams:
        reader = csv.reader(fp, **fmtparams)
    else:
        dialect = csv.Sniffer().sniff(fp.read(1024))
        fp.seek(0)
        reader = csv.reader(fp, dialect)

    table = PrettyTable(**kwargs)
    if field_names:
        table.field_names = field_names
    else:
        table.field_names = [x.strip() for x in next(reader)]

    for row in reader:
        table.add_row([x.strip() for x in row])

    return table


def from_db_cursor(cursor, **kwargs):
    if cursor.description:
        table = PrettyTable(**kwargs)
        table.field_names = [col[0] for col in cursor.description]
        for row in cursor.fetchall():
            table.add_row(row)
        return table


def from_json(json_string, **kwargs):
    table = PrettyTable(**kwargs)
    objects = json.loads(json_string)
    table.field_names = objects[0]
    for obj in objects[1:]:
        row = [obj[key] for key in table.field_names]
        table.add_row(row)
    return table


class TableHandler(HTMLParser):
    def __init__(self, **kwargs):
        HTMLParser.__init__(self)
        self.kwargs = kwargs
        self.tables = []
        self.last_row = []
        self.rows = []
        self.max_row_width = 0
        self.active = None
        self.last_content = ""
        self.is_last_row_header = False
        self.colspan = 0

    def handle_starttag(self, tag, attrs):
        self.active = tag
        if tag == "th":
            self.is_last_row_header = True
        for (key, value) in attrs:
            if key == "colspan":
                self.colspan = int(value)

    def handle_endtag(self, tag):
        if tag in ["th", "td"]:
            stripped_content = self.last_content.strip()
            self.last_row.append(stripped_content)
            if self.colspan:
                for i in range(1, self.colspan):
                    self.last_row.append("")
                self.colspan = 0

        if tag == "tr":
            self.rows.append((self.last_row, self.is_last_row_header))
            self.max_row_width = max(self.max_row_width, len(self.last_row))
            self.last_row = []
            self.is_last_row_header = False
        if tag == "table":
            table = self.generate_table(self.rows)
            self.tables.append(table)
            self.rows = []
        self.last_content = " "
        self.active = None

    def handle_data(self, data):
        self.last_content += data

    def generate_table(self, rows):
        """
        Generates from a list of rows a PrettyTable object.
        """
        table = PrettyTable(**self.kwargs)
        for row in self.rows:
            if len(row[0]) < self.max_row_width:
                appends = self.max_row_width - len(row[0])
                for i in range(1, appends):
                    row[0].append("-")

            if row[1]:
                self.make_fields_unique(row[0])
                table.field_names = row[0]
            else:
                table.add_row(row[0])
        return table

    def make_fields_unique(self, fields):
        """
        iterates over the row and make each field unique
        """
        for i in range(0, len(fields)):
            for j in range(i + 1, len(fields)):
                if fields[i] == fields[j]:
                    fields[j] += "'"


def from_html(html_code, **kwargs):
    """
    Generates a list of PrettyTables from a string of HTML code. Each <table> in
    the HTML becomes one PrettyTable object.
    """

    parser = TableHandler(**kwargs)
    parser.feed(html_code)
    return parser.tables


def from_html_one(html_code, **kwargs):
    """
    Generates a PrettyTables from a string of HTML code which contains only a
    single <table>
    """

    tables = from_html(html_code, **kwargs)
    try:
        assert len(tables) == 1
    except AssertionError:
        raise ValueError(
            "More than one <table> in provided HTML code. Use from_html instead."
        )
    return tables[0]
