#pragma once

#include <heimdall/dataset_view.hpp>

namespace deeplake_api {

using dataset_view = heimdall::dataset_view;
using dataset_view_ptr = std::shared_ptr<dataset_view>;

[[nodiscard]] async::promise<std::shared_ptr<dataset_view>> from_json(const icm::const_json& js);

} // namespace deeplake_api
