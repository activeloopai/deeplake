#include "py_storage.hpp"
#include "spdlog/spdlog.h"

#include <fstream>
#include <utility>

namespace py_api {

    std::string correct_path(const std::string &path) {
        std::string result = path;
        if (result.find('/') == 0) {
            result = result.substr(1);
        }
        return result;
    }

    py_storage::py_storage(pybind11::object obj) : _wrapped_storage(std::move(obj)) {

    }

    ::storage::file_ref py_storage::file(const std::string &path) const {
        try {
            long bytes = _wrapped_storage.attr("get_object_size")(correct_path(path)).cast<long>();
            return {path, bytes};
        } catch (const pybind11::error_already_set &e) {
            spdlog::debug("File does not exist: {}", path);
//            spdlog::debug("Caught exception: {}", e.what());
            return {path, -1};
        }

    }

    std::vector<::storage::file_ref> py_storage::list_files(const std::string &base_dir) const {
        std::vector<::storage::file_ref> files {};

        for (auto file_name: _wrapped_storage.attr("__iter__")()) {
            files.push_back(file(file_name.cast<std::string>()));
        }

        return files;
    }

    std::vector<uint8_t> py_storage::get_bytes(const std::string &path) const {
        auto bytes = _wrapped_storage.attr("get_bytes")(correct_path(path)).cast<pybind11::bytes>();
        std::vector<uint8_t> result {};
        for (auto byte: bytes) {
            result.push_back(byte.cast<uint8_t>());
        }

        spdlog::debug("Read {} bytes from path {}", result.size(), path);

        return result;

    }

    void py_storage::set_bytes(const std::string &path, const std::string &data) const {
        spdlog::debug("Writing {} bytes to path {}", data.size(), path);

        _wrapped_storage.attr("set_bytes")(correct_path(path), pybind11::bytes(data));
        _wrapped_storage.attr("flush")();
    }

}