#pragma once

#include <async/run.hpp>
#include <heimdall/row_view.hpp>

namespace deeplake_api {

using row_view = heimdall::row_view;

[[nodiscard]] inline auto row_view_to_string(const row_view& r)
{
    return async::run_on_main([&r]() {
        return r.to_string();
    });
}

} // namespace deeplake_api