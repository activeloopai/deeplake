#pragma once

#include "../array.hpp"
#include <memory>
#include "../array_to_json.hpp"

namespace nd::impl {

template <typename M>
struct std_map_dict
{
    explicit std_map_dict(M&& data)
        : data_(std::move(data))
    {
    }

    std::vector<std::string> keys() const noexcept
    {
        std::vector<std::string> keys;
        keys.reserve(data_.size());
        for (const auto& [key, _] : data_) {
            keys.push_back(key);
        }
        return keys;
    }

    bool contains(const std::string& key) const
    {
        if constexpr (impl::has_contains_member_function_v<M>) {
            return data_.contains(key);
        } else {
            auto ks = keys();
            return std::find(ks.begin(), ks.end(), key) != ks.end();
        }
    }

    array operator[](const std::string& key) const
    {
        auto it = data_.find(key);
        if (it == data_.end()) {
            throw std::out_of_range("Key not found");
        }
        return it->second;
    }

    dict eval() const
    {
        icm::string_map<icm::const_json> result;
        for (const auto& [key, value] : data_) {
            result.emplace(key, array_to_json(value));
        }
        return dict(icm::const_json(std::move(result)));
    }

private:
    M data_;
};

} // namespace nd::impl
