#include <pybind11/stl.h>
#include "deeplake.hpp"
#include "deeplog/deeplog.hpp"
#include "deeplog/actions/actions.hpp"

PYBIND11_MAKE_OPAQUE(std::optional<std::string>);

PYBIND11_MODULE(_deeplake, mod_deeplake) {
    auto mod_deeplog = mod_deeplake.def_submodule("_deeplog");
    auto mod_actions = mod_deeplog.def_submodule("_actions");

    py_api::deeplake::pybind(mod_deeplake);
    py_api::deeplog::pybind(mod_deeplog);
    py_api::actions::pybind(mod_actions);

}
