#pragma once

#include <nlohmann/json.hpp>
#include <arrow/api.h>

namespace deeplog {

    class action {

    public:
        virtual nlohmann::json to_json() = 0;

        virtual std::string action_name() = 0;

    };


}
