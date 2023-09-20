#pragma once

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../../../deeplog/actions/action.hpp"

namespace py_api {

    class actions {

    public:
        static void pybind(pybind11::module &);
    };
}
