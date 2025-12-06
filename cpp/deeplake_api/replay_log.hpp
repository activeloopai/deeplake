#pragma once
#include <async/promise.hpp>
#include <icm/string_map.hpp>
#include <optional>
#include <string>

namespace deeplake_api {
[[nodiscard]] async::promise<void> replay_log(const std::string& source_path,
                                              const std::string& destination_path,
                                              icm::string_map<>&& src_creds = {},
                                              icm::string_map<>&& dst_creds = {},
                                              std::optional<std::string> token = std::nullopt);
}
