#include "actions.hpp"
#include "../../../deeplog/actions/add_file_action.hpp"
#include "../../../deeplog/actions/create_branch_action.hpp"
#include "../../../deeplog/actions/metadata_action.hpp"
#include "../../../deeplog/actions/protocol_action.hpp"
#include "../../../deeplog/actions/remove_file_action.hpp"
#include "../../../deeplog/actions/create_commit_action.hpp"
#include "../../../deeplog/actions/create_tensor_action.hpp"
#include <variant>
#include <iostream>

namespace py_api {

    std::shared_ptr<deeplog::create_tensor_action> create_tensor_action(std::string id,
                                                                        std::string name,
                                                                        std::optional<std::string> dtype,
                                                                        std::string htype,
                                                                        const long &length,
                                                                        const bool &is_link,
                                                                        const bool &is_sequence,
                                                                        const bool &hidden,
                                                                        const std::optional<std::string> &chunk_compression,
                                                                        const std::optional<std::string> &sample_compression,
                                                                        const std::map<std::string, std::map<std::string, pybind11::object>> &links,
                                                                        const std::optional<long> &max_chunk_size,
                                                                        const std::vector<unsigned long> &min_shape,
                                                                        const std::vector<unsigned long> &max_shape,
                                                                        const std::optional<long> &tiling_threshold,
                                                                        const std::optional<std::string> &typestr,
                                                                        const bool &verify,
                                                                        std::string version) {

        std::map<std::string, deeplog::tensor_link> links_map{};
        for (const auto &link: links) {
            std::optional<bool> flatten_sequence;
            std::optional<std::string> extend;
            std::optional<std::string> update;

            if (link.second.contains("flatten_sequence")) {
                flatten_sequence = link.second.at("flatten_sequence").cast<bool>();
            }
            if (link.second.contains("extend")) {
                extend = link.second.at("extend").cast<std::string>();
            }  if (link.second.contains("update")) {
                update = link.second.at("update").cast<std::string>();
            }
            links_map.insert({link.first, deeplog::tensor_link(extend,
                                                               flatten_sequence,
                                                               update)});
        }

        return std::make_shared<deeplog::create_tensor_action>(std::move(id), std::move(name), std::move(dtype), std::move(htype), length, is_link, is_sequence, hidden,
                                                               chunk_compression, sample_compression, std::move(links_map), max_chunk_size,
                                                               min_shape, max_shape, tiling_threshold, typestr, verify, std::move(version));
    }

    void actions::pybind(pybind11::module &module) {
        pybind11::class_<deeplog::action, std::shared_ptr<deeplog::action>>(module, "DeepLogAction");

        pybind11::class_<deeplog::add_file_action, deeplog::action, std::shared_ptr<deeplog::add_file_action>>(module, "AddFileAction")
                .def(pybind11::init<std::string, std::string, long, long, bool, long>(),
                     pybind11::arg("path"), pybind11::arg("type"), pybind11::arg("size"), pybind11::arg("modification_time"), pybind11::arg("data_change"), pybind11::arg("num_samples"))
                .def_readonly("path", &deeplog::add_file_action::path)
                .def_readonly("type", &deeplog::add_file_action::type)
                .def_readonly("size", &deeplog::add_file_action::size)
                .def_readonly("modification_time", &deeplog::add_file_action::modification_time)
                .def_readonly("num_samples", &deeplog::add_file_action::num_samples);

        pybind11::class_<deeplog::create_branch_action, deeplog::action, std::shared_ptr<deeplog::create_branch_action>>(module, "CreateBranchAction")
                .def(pybind11::init<std::string, std::string, std::string, unsigned long>(),
                     pybind11::arg("id"), pybind11::arg("name"), pybind11::arg("from_id"), pybind11::arg("from_version"))
                .def_readonly("id", &deeplog::create_branch_action::id)
                .def_readonly("name", &deeplog::create_branch_action::name)
                .def_readonly("from_id", &deeplog::create_branch_action::from_id)
                .def_readonly("from_version", &deeplog::create_branch_action::from_version);


        pybind11::class_<deeplog::create_commit_action, deeplog::action, std::shared_ptr<deeplog::create_commit_action>>(module, "CreateCommitAction")
                .def(pybind11::init<std::string, std::string, unsigned long, std::optional<std::string>, long>(),
                     pybind11::arg("id"), pybind11::arg("branch_id"), pybind11::arg("branch_version"), pybind11::arg("message"), pybind11::arg("commit_time"))
                .def_readonly("id", &deeplog::create_commit_action::id)
                .def_readonly("branch_id", &deeplog::create_commit_action::branch_id)
                .def_readonly("branch_version", &deeplog::create_commit_action::branch_version)
                .def_readonly("message", &deeplog::create_commit_action::message)
                .def_readonly("commit_time", &deeplog::create_commit_action::commit_time);

        pybind11::class_<deeplog::create_tensor_action, deeplog::action, std::shared_ptr<deeplog::create_tensor_action>>(module, "CreateTensorAction")
                .def(pybind11::init(&create_tensor_action),
                     pybind11::arg("id"),
                     pybind11::arg("name"),
                     pybind11::arg("dtype"),
                     pybind11::arg("htype"),
                     pybind11::arg("length"),
                     pybind11::arg("is_link"),
                     pybind11::arg("is_sequence"),
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
                .def_readonly("id", &deeplog::create_tensor_action::id)
                .def_readonly("name", &deeplog::create_tensor_action::name)
                .def_readonly("dtype", &deeplog::create_tensor_action::dtype)
                .def_readonly("htype", &deeplog::create_tensor_action::htype)
                .def_readonly("length", &deeplog::create_tensor_action::length)
                .def_readonly("is_link", &deeplog::create_tensor_action::is_link)
                .def_readonly("is_sequence", &deeplog::create_tensor_action::is_sequence)
                .def_readonly("hidden", &deeplog::create_tensor_action::hidden)
                .def_readonly("chunk_compression", &deeplog::create_tensor_action::chunk_compression)
                .def_readonly("sample_compression", &deeplog::create_tensor_action::sample_compression)
                .def_readonly("links", &deeplog::create_tensor_action::links)
                .def_readonly("max_chunk_size", &deeplog::create_tensor_action::max_chunk_size)
                .def_readonly("min_shape", &deeplog::create_tensor_action::min_shape)
                .def_readonly("max_shape", &deeplog::create_tensor_action::max_shape)
                .def_readonly("tiling_threshold", &deeplog::create_tensor_action::tiling_threshold)
                .def_readonly("typestr", &deeplog::create_tensor_action::typestr)
                .def_readonly("verify", &deeplog::create_tensor_action::verify)
                .def_readonly("version", &deeplog::create_tensor_action::version);

        pybind11::class_<deeplog::metadata_action, deeplog::action, std::shared_ptr<deeplog::metadata_action>>(module, "MetadataAction")
                .def(pybind11::init<std::string, std::string, std::string, long>(),
                     pybind11::arg("id"), pybind11::arg("name"), pybind11::arg("description"), pybind11::arg("created_time"))
                .def_readonly("id", &deeplog::metadata_action::id)
                .def_readonly("name", &deeplog::metadata_action::name)
                .def_readonly("description", &deeplog::metadata_action::description)
                .def_readonly("created_time", &deeplog::metadata_action::created_time);

        pybind11::class_<deeplog::protocol_action, deeplog::action, std::shared_ptr<deeplog::protocol_action>>(module, "ProtocolAction")
                .def(pybind11::init<int, int>(),
                     pybind11::arg("min_reader_version"), pybind11::arg("min_writer_version"))
                .def_readonly("min_reader_version", &deeplog::protocol_action::min_reader_version)
                .def_readonly("min_writer_version", &deeplog::protocol_action::min_writer_version);

        pybind11::class_<deeplog::remove_file_action, deeplog::action, std::shared_ptr<deeplog::remove_file_action>>(module, "RemoveFileAction")
                .def(pybind11::init<std::string, long, long, bool>(),
                     pybind11::arg("path"), pybind11::arg("size"), pybind11::arg("deletion_timestamp"), pybind11::arg("data_change"))
                .def_readonly("path", &deeplog::remove_file_action::path)
                .def_readonly("size", &deeplog::remove_file_action::size)
                .def_readonly("deletion_timestamp", &deeplog::remove_file_action::deletion_time)
                .def_readonly("data_change", &deeplog::remove_file_action::data_change);

        pybind11::class_<deeplog::tensor_link, std::shared_ptr<deeplog::tensor_link>>(module, "TensorLink")
                .def(pybind11::init<std::string, std::optional<bool>, std::string>(),
                     pybind11::arg("extend"), pybind11::arg("flatten_sequence"), pybind11::arg("update"))
                .def_readonly("extend", &deeplog::tensor_link::extend)
                .def_readonly("flatten_sequence", &deeplog::tensor_link::flatten_sequence)
                .def_readonly("update", &deeplog::tensor_link::update);
    }
}
