#include "deeplake.hpp"
#include <iostream>
#include <spdlog/spdlog.h>

namespace py_api {

    void deeplake::pybind(pybind11::module &module) {
        pybind11::class_<::py_api::logger, std::shared_ptr<::py_api::logger >>(module, "Logger")
                .def_static("set_log_level", &::py_api::logger::set_log_level)
                .def_static("debug", &::py_api::logger::debug)
                .def_static("info", &::py_api::logger::info)
                .def_static("warn", &::py_api::logger::warn)
                .def_static("err", &::py_api::logger::err)
                .def_static("critical", &::py_api::logger::critical);
    }

    void logger::set_log_level(const std::string &level) {
        auto lc_level = level;
        std::transform(lc_level.begin(), lc_level.end(), lc_level.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (lc_level == "debug") {
            spdlog::set_level(spdlog::level::debug);
        } else if (lc_level == "info") {
            spdlog::set_level(spdlog::level::info);
        } else if (lc_level == "warn") {
            spdlog::set_level(spdlog::level::warn);
        } else if (lc_level == "err") {
            spdlog::set_level(spdlog::level::err);
        } else if (lc_level == "critical") {
            spdlog::set_level(spdlog::level::critical);
        } else if (lc_level == "off") {
            spdlog::set_level(spdlog::level::off);
        }

        spdlog::debug("Set log level to {}", level);
    }

    void logger::debug(const std::string &msg) {
        spdlog::debug(msg);
    }

    void logger::info(const std::string &msg) {
        spdlog::info(msg);
    }

    void logger::warn(const std::string &msg) {
        spdlog::warn(msg);
    }

    void logger::err(const std::string &msg) {
        spdlog::error(msg);
    }

    void logger::critical(const std::string &msg) {
        spdlog::critical(msg);
    }
}
