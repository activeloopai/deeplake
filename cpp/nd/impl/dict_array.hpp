#pragma once

#include "../adapt.hpp"
#include "../dtype.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"

#include <icm/string_map.hpp>

#include <string>

namespace nd::impl {

class dict_array
{
public:
    explicit dict_array(icm::string_map<nd::array> dict, uint64_t num_rows = 1)
        : dict_(std::move(dict))
        , num_rows_(num_rows)
    {
        ASSERT(!dict_.empty());
    }

    enum dtype dtype() const noexcept
    {
        return nd::dtype::object;
    }

    nd::array get(int64_t index) const
    {
        if (num_rows_ == 1) {
            icm::string_map<nd::array> key2arr;
            for (const auto& [key, arr] : dict_) {
                key2arr[key] = arr[index];
            }
            return adapt(nd::dict(std::move(key2arr)));
        }
        std::vector<nd::array> results;
        for (auto idx = 0; idx < num_rows_; ++idx) {
            icm::string_map<nd::array> key2arr;
            for (const auto& [key, arr] : dict_) {
                key2arr[key] = arr[index][idx];
            }
            results.emplace_back(adapt(nd::dict(std::move(key2arr))));
        }
        return nd::dynamic(std::move(results));
    }

    nd::dict value(int64_t index) const;

    icm::shape shape() const noexcept
    {
        return dict_.begin()->second.shape();
    }

    uint8_t dimensions() const
    {
        return static_cast<uint8_t>(dict_.begin()->second.dimensions());
    }

    constexpr bool is_dynamic() const noexcept
    {
        return true;
    }

private:
    uint64_t num_rows_ = 1;
    icm::string_map<nd::array> dict_;
};

class dict_array_entry
{
public:
    dict_array_entry(std::shared_ptr<icm::string_map<nd::array>> dict, int64_t index)
        : dict_(std::move(dict))
        , index_(index)
    {
    }

    std::vector<std::string> keys() const noexcept
    {
        std::vector<std::string> keys;
        keys.reserve(dict_->size());
        for (const auto& [key, _] : *dict_) {
            keys.push_back(key);
        }
        return keys;
    }

    bool contains(const std::string& key) const
    {
        return dict_->contains(key);
    }

    array operator[](const std::string& key) const
    {
        auto it = dict_->find(key);
        if (it == dict_->end()) {
            throw std::out_of_range("Key not found");
        }
        return switch_dtype(it->second.dtype(), [this, it]<typename T>() mutable {
            if constexpr (std::is_same_v<T, dict>) {
                return adapt(it->second.dict_value(index_));
            } else if constexpr (std::is_same_v<T, std::string_view>) {
                return adapt(std::string(it->second.value<std::string_view>(index_)));
            } else {
                return adapt(it->second.value<T>(index_));
            }
        });
    }

private:
    std::shared_ptr<icm::string_map<nd::array>> dict_;
    int64_t index_;
};

inline dict dict_array::value(int64_t index) const
{
    return dict(dict_array_entry(std::make_shared<icm::string_map<nd::array>>(dict_), index));
}

} // namespace nd::impl
