#pragma once

#include <icm/json.hpp>
#include <icm/const_json.hpp>

#include <chrono>
#include <string>
#include <variant>

namespace deeplake_api {

struct log_entry
{
    std::string session_id;
    int64_t timestamp; // Microseconds since epoch
    std::string operation;
    std::variant<icm::json, icm::const_json> args;  // json for write path, const_json for read path

    [[nodiscard]] icm::json to_json() const
    {
        icm::json j;
        j["session_id"] = session_id;
        j["timestamp"] = timestamp;
        j["operation"] = operation;
        // Extract json from variant for serialization
        if (std::holds_alternative<icm::json>(args)) {
            j["args"] = std::get<icm::json>(args);
        } else {
            // Convert const_json to json for serialization
            std::string args_str = std::get<icm::const_json>(args).dump();
            j["args"] = icm::json::parse(args_str);
        }
        return j;
    }

    static log_entry from_json(const icm::const_json& j)
    {
        log_entry entry;
        entry.session_id = j["session_id"].get<std::string>();
        entry.timestamp = j["timestamp"].get<int64_t>();
        entry.operation = j["operation"].get<std::string>();
        // Store as const_json for read path - no conversion needed
        entry.args = j["args"];
        return entry;
    }
};

} // namespace deeplake_api
