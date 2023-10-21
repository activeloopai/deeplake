#pragma once

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py_api {
    class logger {
    public:
        static void set_log_level(const std::string &level);

        static void debug(const std::string &msg);

        static void info(const std::string &msg);

        static void warn(const std::string &msg);

        static void err(const std::string &msg);

        static void critical(const std::string &msg);

    };

    class deeplake {

    public:
        static void pybind(pybind11::module &);
    };
}
