#include <pybind11/pybind11.h>
#include "../deeplog/deeplog.hpp"

PYBIND11_MODULE(_deeplake, m) {
    auto m_deeplog = m.def_submodule("_deeplog");

    pybind11::class_<deeplake::deeplog>(m_deeplog, "DeepLogCpp")
            .def_static("open", &deeplake::deeplog::open)
            .def_static("create", &deeplake::deeplog::create)
//            .def("version", &deeplake::deeplog::version)
            .def("path", &deeplake::deeplog::path);
}