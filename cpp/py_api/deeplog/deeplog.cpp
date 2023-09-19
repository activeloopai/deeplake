#include <pybind11/stl.h>
#include "../../deeplog/deeplog.hpp"
#include "deeplog.hpp"

namespace py_api {
    void deeplog::pybind(pybind11::module &module) {
        pybind11::class_<::deeplog::deeplog_state>(module, "DeepLogState")

        pybind11::class_<::deeplog::deeplog>(module, "DeepLogCpp")
                .def_static("open", &::deeplog::deeplog::open)
                .def_static("create", &::deeplog::deeplog::create)
                .def("version", &::deeplog::deeplog::version)
                .def("path", &::deeplog::deeplog::path)
                .def("protocol", &::deeplog::deeplog::protocol)
                .def("metadata", &::deeplog::deeplog::metadata)
                .def("branches", &::deeplog::deeplog::branches)
                .def("branch_by_id", &::deeplog::deeplog::branch_by_id)
                .def("data_files", &::deeplog::deeplog::data_files)
                .def("commit", &::deeplog::deeplog::commit)
                .def("checkpoint", &::deeplog::deeplog::checkpoint)
                ;

    }
}
