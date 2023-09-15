#include <pybind11/pybind11.h>
#include "deeplog.hpp"

namespace deeplake {
    PYBIND11_MODULE(deeplog, m) {
        pybind11::class_<deeplog>(m, "deeplog")
                .def_static("create", &deeplog::create)
                .def_static("open", &deeplog::open)
                .def("path", &deeplog::path);
    }
}