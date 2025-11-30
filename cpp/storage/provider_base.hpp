#pragma once

#include <async/promise.hpp>
#include <base/atfork.hpp>
#include <icm/string_map.hpp>

#include <string>
#include <utility>

namespace storage {

class provider_base : public base::atfork_handler
{
public:
    using params_t = icm::string_map<>;
    using params_updater_t = std::function<async::promise<params_t>(const params_t&, const std::string&)>;

public:
    virtual params_t params() const = 0;

    virtual std::string url() const noexcept = 0;

public:
    virtual ~provider_base() noexcept = default;

    std::string original_url() const noexcept
    {
        auto ps = params();
        auto it = ps.find("original_url");
        return it != ps.end() ? it->second : url();
    }

    std::string token() const noexcept
    {
        auto ps = params();
        auto it = ps.find("token");
        return it != ps.end() ? it->second : "";
    }

    std::pair<std::string, params_t> serialize() const
    {
        return {url(), params()};
    }

protected:
    /**
     * @brief Being called to let provider to update underlying credentials, fetch client, etc.
     */
    [[nodiscard]] virtual async::promise<void> update() const = 0;
};

} // namespace storage
