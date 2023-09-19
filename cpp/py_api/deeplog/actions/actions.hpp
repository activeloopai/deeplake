#pragma once

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../../../deeplog/actions/action.hpp"

namespace py_api {

    class py_deeplog_action : public ::deeplog::action {
    public:
        using ::deeplog::action::action;

        void to_json(nlohmann::json &json) override {

        }

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override {
            return arrow::Status::OK();
        }
    };

    class actions {

    public:
        static void pybind(pybind11::module &);
    };
}
