#include <pybind11/stl.h>
#include "../../deeplog/deeplog.hpp"
#include "deeplog.hpp"
#include "../../deeplog/snapshot.hpp"
#include "../../deeplog/metadata_snapshot.hpp"
#include "../../deeplog/deeplog_v3.hpp"
#include "../../deeplog/optimistic_transaction.hpp"

namespace py_api {
    void deeplog::pybind(pybind11::module &module) {
        pybind11::class_<::deeplog::deeplog, std::shared_ptr<::deeplog::deeplog>>(module, "DeepLog")
                .def_static("open", &::deeplog::deeplog::open)
                .def_static("create", &::deeplog::deeplog::create)
                .def_readonly("path", &::deeplog::deeplog::path)
                .def("log_format", &::deeplog::deeplog::log_format)
                .def("version", &::deeplog::deeplog::version)
                .def("checkpoint", &::deeplog::deeplog::checkpoint);

        pybind11::class_<::deeplog::deeplog_v3, ::deeplog::deeplog, std::shared_ptr<::deeplog::deeplog_v3>>(module, "DeepLogV3")
                .def("log_format", &::deeplog::deeplog::log_format);

        pybind11::class_<::deeplog::snapshot, std::shared_ptr<::deeplog::snapshot>>(module, "Snapshot")
                .def(pybind11::init<std::string, const std::shared_ptr<::deeplog::deeplog> &>(),
                     pybind11::arg("branch_id"), pybind11::arg("deeplog"))
                .def(pybind11::init<std::string, const long &, const std::shared_ptr<::deeplog::deeplog> &>(),
                     pybind11::arg("branch_id"), pybind11::arg("version"), pybind11::arg("deeplog"))
                .def("data_files", &::deeplog::snapshot::data_files)
                .def("commits", &::deeplog::snapshot::commits)
                .def("tensors", &::deeplog::snapshot::tensors)
                .def_readonly("version", &::deeplog::snapshot::version)
                .def_readonly("branch_id", &::deeplog::snapshot::branch_id);

        pybind11::class_<::deeplog::metadata_snapshot, std::shared_ptr<::deeplog::metadata_snapshot>>(module, "MetadataSnapshot")
                .def(pybind11::init<const std::shared_ptr<::deeplog::deeplog> &>(),
                     pybind11::arg("deeplog"))
                .def(pybind11::init<const long &, const std::shared_ptr<::deeplog::deeplog> &>(),
                     pybind11::arg("version"), pybind11::arg("deeplog"))
                .def("protocol", &::deeplog::metadata_snapshot::protocol)
                .def("metadata", &::deeplog::metadata_snapshot::metadata)
                .def("branches", &::deeplog::metadata_snapshot::branches)
                .def("branch_by_id", &::deeplog::metadata_snapshot::branch_by_id)
                .def("branch_id", &::deeplog::metadata_snapshot::branch_id)
                .def_readonly("version", &::deeplog::metadata_snapshot::version);

        pybind11::class_<::deeplog::optimistic_transaction>(module, "OptimisticTransaction")
                .def(pybind11::init<const std::shared_ptr<::deeplog::snapshot> &>(),
                     pybind11::arg("snapshot"))
                .def("add", &::deeplog::optimistic_transaction::add, pybind11::arg("action"))
                .def("commit", &::deeplog::optimistic_transaction::commit)
                .def_readonly("snapshot", &::deeplog::optimistic_transaction::snapshot);
    }
}
