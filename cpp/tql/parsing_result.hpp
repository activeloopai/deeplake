#pragma once

#include "input.hpp"

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <unordered_set>

namespace tql {

struct parsing_result
{
    template <typename F>
    requires (std::is_invocable_v<F, const parsing_result&>)
    auto traverse(F f) const
    {
        if (from.index() == 1) {
            const auto& rs = std::get<1>(from);
            for (const auto& r: rs) {
                f(*r);
            }
        }
        f(*this);
    }

    inline std::vector<std::string> ds_input_tensors() const
    {
        std::vector<std::string> result;
        std::unordered_set<std::string> seen;
        result.reserve(input_data.tensors.size());
        for (const auto& t : input_data.tensors) {
            if (seen.emplace(t.name).second) {
                result.push_back(t.name);
            }
        }
        return result;
    }

    std::variant<std::monostate, std::vector<std::shared_ptr<parsing_result>>, std::string> from;
    input input_data;
    std::vector<std::string> ds_output_tensors;
    bool is_filter = true;
};

}
