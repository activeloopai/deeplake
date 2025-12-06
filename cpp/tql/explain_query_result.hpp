#pragma once

#include <icm/const_json.hpp>

#include <string>

namespace tql {

class explain_query_result
{
public:
    explicit explain_query_result(std::string result, icm::const_json result_json)
        : result_(std::move(result))
        , result_json_(std::move(result_json))
    {
    }

    inline const std::string& get_result() const noexcept
    {
        return result_;
    }

    inline const auto& get_result_json() const noexcept
    {
        return result_json_;
    }

private:
    std::string result_;
    icm::const_json result_json_;
};

} // namespace tql