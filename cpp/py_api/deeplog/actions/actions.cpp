#include "actions.hpp"
#include "../../../deeplog/actions/add_file_action.hpp"
#include "../../../deeplog/actions/create_branch_action.hpp"
#include "../../../deeplog/actions/metadata_action.hpp"
#include "../../../deeplog/actions/protocol_action.hpp"
#include "../../../deeplog/actions/remove_file_action.hpp"
#include "../../../deeplog/actions/create_commit_action.hpp"
#include "../../../deeplog/actions/create_tensor_action.hpp"

namespace py_api {

    void actions::pybind(pybind11::module &module) {
        auto base_action = pybind11::class_<py_deeplog_action>(module, "DeepLogAction")
                .def(pybind11::init<>());

        pybind11::class_<deeplog::add_file_action>(module, "AddFileAction", base_action)
                .def(pybind11::init<std::string, long, long, bool>(),
                     pybind11::arg("path"), pybind11::arg("size"), pybind11::arg("modification_time"), pybind11::arg("data_change"))
                .def("path", &deeplog::add_file_action::path)
                .def("size", &deeplog::add_file_action::size)
                .def("modification_time", &deeplog::add_file_action::modification_time);

        pybind11::class_<deeplog::create_branch_action>(module, "CreateBranchAction", base_action)
                .def(pybind11::init<std::string, std::string, std::string, long>(),
                     pybind11::arg("id"), pybind11::arg("name"), pybind11::arg("from_branch_id"), pybind11::arg("from_branch_version"))
                .def("id", &deeplog::create_branch_action::id)
                .def("name", &deeplog::create_branch_action::name)
                .def("from_id", &deeplog::create_branch_action::from_id)
                .def("from_version", &deeplog::create_branch_action::from_version);


        pybind11::class_<deeplog::create_commit_action>(module, "CreateCommitAction", base_action)
                .def(pybind11::init<std::string, std::string, long, std::optional<std::string>, long>(),
                     pybind11::arg("id"), pybind11::arg("branch_id"), pybind11::arg("branch_version"), pybind11::arg("message"), pybind11::arg("commit_time"))
                .def("id", &deeplog::create_commit_action::id)
                .def("branch_id", &deeplog::create_commit_action::branch_id)
                .def("branch_version", &deeplog::create_commit_action::branch_version)
                .def("message", &deeplog::create_commit_action::message)
                .def("commit_time", &deeplog::create_commit_action::commit_time);

        pybind11::class_<deeplog::create_tensor_action>(module, "CreateTensorAction", base_action)
                .def(pybind11::init<std::string, std::string, std::string, std::string, long, bool, bool, bool, std::optional<std::string>,
                             std::optional<std::string>,
                             std::map<std::string, std::map<std::string, std::string>>,
                             std::optional<long>,
                             std::vector<long>,
                             std::vector<long>,
                             std::optional<long>, std::optional<std::string>, bool, std::string>(),
                     pybind11::arg("id"),
                     pybind11::arg("name"),
                     pybind11::arg("dtype"),
                     pybind11::arg("htype"),
                     pybind11::arg("length"),
                     pybind11::arg("link"),
                     pybind11::arg("sequence"),
                     pybind11::arg("hidden"),
                     pybind11::arg("chunk_compression"),
                     pybind11::arg("sample_compression"),
                     pybind11::arg("links"),
                     pybind11::arg("max_chunk_size"),
                     pybind11::arg("min_shape"),
                     pybind11::arg("max_shape"),
                     pybind11::arg("tiling_threshold"),
                     pybind11::arg("typestr"),
                     pybind11::arg("verify"),
                     pybind11::arg("version")
                )
                .def("id", &deeplog::create_tensor_action::id)
                .def("name", &deeplog::create_tensor_action::name)
                .def("dtype", &deeplog::create_tensor_action::dtype)
                .def("htype", &deeplog::create_tensor_action::htype)
                .def("length", &deeplog::create_tensor_action::length)
                .def("link", &deeplog::create_tensor_action::link)
                .def("sequence", &deeplog::create_tensor_action::sequence)
                .def("hidden", &deeplog::create_tensor_action::hidden)
                .def("chunk_compression", &deeplog::create_tensor_action::chunk_compression)
                .def("sample_compression", &deeplog::create_tensor_action::sample_compression)
                .def("links", &deeplog::create_tensor_action::links)
                .def("max_chunk_size", &deeplog::create_tensor_action::max_chunk_size)
                .def("min_shape", &deeplog::create_tensor_action::min_shape)
                .def("max_shape", &deeplog::create_tensor_action::max_shape)
                .def("tiling_threshold", &deeplog::create_tensor_action::tiling_threshold)
                .def("typestr", &deeplog::create_tensor_action::typestr)
                .def("verify", &deeplog::create_tensor_action::verify)
                .def("version", &deeplog::create_tensor_action::version);

        pybind11::class_<deeplog::metadata_action>(module, "MetadataAction", base_action)
                .def(pybind11::init<std::string, std::string, std::string, long>(),
                     pybind11::arg("id"), pybind11::arg("name"), pybind11::arg("description"), pybind11::arg("created_time"))
                .def("id", &deeplog::metadata_action::id)
                .def("name", &deeplog::metadata_action::name)
                .def("description", &deeplog::metadata_action::description)
                .def("created_time", &deeplog::metadata_action::created_time);

        pybind11::class_<deeplog::protocol_action>(module, "ProtocolAction", base_action)
                .def(pybind11::init<int, int>(),
                     pybind11::arg("min_reader_version"), pybind11::arg("min_writer_version"))
                .def("min_reader_version", &deeplog::protocol_action::min_reader_version)
                .def("min_writer_version", &deeplog::protocol_action::min_writer_version);

        pybind11::class_<deeplog::remove_file_action>(module, "RemoveFileAction", base_action)
                .def(pybind11::init<std::string, long, long, bool>(),
                     pybind11::arg("path"), pybind11::arg("size"), pybind11::arg("deletion_timestamp"), pybind11::arg("data_change"))
                .def("path", &deeplog::remove_file_action::path)
                .def("size", &deeplog::remove_file_action::size)
                .def("deletion_timestamp", &deeplog::remove_file_action::deletion_timestamp)
                .def("data_change", &deeplog::remove_file_action::data_change);
    }
}
