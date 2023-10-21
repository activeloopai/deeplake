#pragma once

#include <pybind11/pybind11.h>
#include <fstream>
#include "../../storage/storage.hpp"
#include "../../storage/file_ref.hpp"
#include "spdlog/spdlog.h"

namespace py_api {

    class py_storage : public ::storage::storage {

    public:
        py_storage(pybind11::object obj);

        ::storage::file_ref file(const std::string &path) const override;

        std::vector<::storage::file_ref> list_files(const std::string &base_dir) const override;

        std::vector<uint8_t> get_bytes(const std::string &path) const override;

        void set_bytes(const std::string &path, const std::string &data) const override;

    private:
        pybind11::object _wrapped_storage;
    };
}