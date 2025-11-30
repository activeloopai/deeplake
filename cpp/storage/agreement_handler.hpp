#pragma once

#include <functional>
#include <string>
#include <vector>

namespace storage {

using agreement_handler = std::function<bool(const std::string&, const std::string&, const std::vector<std::string>&)>;

}
