#pragma once

#include <async/promise.hpp>
#include <async/run.hpp>

#include <icm/json.hpp>

#include <functional>

namespace deeplake_api {
class read_only_metadata
{
public:
    read_only_metadata(std::function<icm::json(const std::string& key)> get_metadata,
                       std::function<std::vector<std::string>()> get_keys)
        : get_metadata_(std::move(get_metadata)), get_keys_(std::move(get_keys)){};

    read_only_metadata(const read_only_metadata&) = delete;
    read_only_metadata& operator=(const read_only_metadata&) = delete;

    read_only_metadata(read_only_metadata&&) = default;
    read_only_metadata& operator=(read_only_metadata&&) = default;

    icm::json get(const std::string& key) const
    {
        return get_metadata_(key);
    }

    std::vector<std::string> keys() const
    {
        return get_keys_();
    }

    bool contains(const std::string& key) const
    {
        auto keys = get_keys_();
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    }

private:
    std::function<icm::json(const std::string& key)> get_metadata_;
    std::function<std::vector<std::string>()> get_keys_;
};

class metadata : public read_only_metadata
{
public:
    metadata(std::function<void(const std::string& key, const icm::json& value)> set_metadata,
             std::function<icm::json(const std::string& key)> get_metadata,
             std::function<std::vector<std::string>()> get_keys)
        : read_only_metadata(std::move(get_metadata), std::move(get_keys)), set_metadata_(std::move(set_metadata)){};

    metadata(const metadata&) = delete;
    metadata& operator=(const metadata&) = delete;

    metadata(metadata&&) = default;
    metadata& operator=(metadata&&) = default;

    void set(const std::string& key, const icm::json& value)
    {
        return set_metadata_(key, value);
    }

private:
    std::function<void(const std::string& key, const icm::json& value)> set_metadata_;
};

[[nodiscard]] inline auto set(const std::shared_ptr<metadata>& m, const std::string& key, const icm::json& value)
{
    return async::run_on_main([m, key, value]() {
        m->set(key, value);
    });
}

} // namespace deeplake_api
