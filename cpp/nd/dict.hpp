#pragma once

/**
 * @brief Declaration and implementation of `dict` class.
 * @file dict.hpp
 */

#include "exceptions.hpp"
#include "json_adapt.hpp"
#include "impl/mpl.hpp"

#include <base/assert.hpp>
#include <base/memory_buffer.hpp>
#include <icm/const_json.hpp>
#include <icm/fifo_map.hpp>
#include <icm/shape.hpp>
#include <icm/string_map.hpp>

#include <map>
#include <unordered_map>
#include <variant>

namespace nd {

// Forward declarations
class array;

/**
 * @brief A class representing a dictionary.
 */

class dict
{
private:
    struct holder_
    {
        holder_(bool has_data, bool has_serialize, bool has_eval)
            : has_data(has_data)
            , has_serialize(has_serialize)
            , has_eval(has_eval)
        {
        }

        virtual ~holder_() = default;

        virtual icm::const_json data() const = 0;
        virtual std::string serialize() const = 0;
        virtual dict eval() const = 0;
        virtual std::vector<std::string> keys() const = 0;
        virtual bool contains(const std::string& key) const = 0;
        virtual array operator[](const std::string& key) const = 0;

        bool has_data;
        bool has_serialize;
        bool has_eval;
    };

    template <typename I>
    struct concrete_holder_ final : public holder_
    {
        static_assert((impl::has_keys_member_function_v<I> && impl::has_subscript_member_operator_v<I>) ||
                          impl::has_dict_data_member_function_v<I>,
                      "The dict adaptor must have keys() and operator[] methods or data() method.");

        explicit concrete_holder_(I&& i)
            : holder_(impl::has_dict_data_member_function_v<I>,
                      impl::has_serialize_member_function_v<I>,
                      impl::has_dict_eval_member_function_v<I>)
            , impl_(std::move(i))
        {
        }

        explicit concrete_holder_(const I& i)
            : holder_(impl::has_dict_data_member_function_v<I>,
                      impl::has_serialize_member_function_v<I>,
                      impl::has_dict_eval_member_function_v<I>)
            , impl_(i)
        {
        }

        icm::const_json data() const override
        {
            if constexpr (impl::has_dict_data_member_function_v<I>) {
                return impl_.data();
            } else {
                throw invalid_operation("data() method is not implemented for this array.");
            }
        }

        std::string serialize() const override
        {
            if constexpr (impl::has_serialize_member_function_v<I>) {
                return impl_.serialize();
            } else {
                throw invalid_operation("serialize() method is not implemented for this array.");
            }
        }

        dict eval() const override;

        array operator[](const std::string& key) const override;

        std::vector<std::string> keys() const override
        {
            if constexpr (impl::has_keys_member_function_v<I>) {
                return impl_.keys();
            } else {
                std::vector<std::string> keys;
                for (const auto& [key, value] : impl_.data()) {
                    keys.push_back(std::string(key));
                }
                return keys;
            }
        }

        bool contains(const std::string& key) const override
        {
            if constexpr (impl::has_contains_member_function_v<I>) {
                return impl_.contains(key);
            } else {
                auto ks = keys();
                return std::find(ks.begin(), ks.end(), key) != ks.end();
            }
        }

        ~concrete_holder_() = default;

    private:
        I impl_;
    };

public:
    dict() = default;

    dict(std::map<std::string, array> m);
    dict(icm::string_map<array> m);
    dict(std::unordered_map<std::string, array> m);
    dict(icm::fifo_map<std::string, array> m);
    dict(icm::const_json v);
    dict(base::memory_buffer b);

    template <typename I>
    requires(!std::is_same_v<dict, std::decay_t<I>>)
    explicit dict(I&& impl)
        : hld_(std::make_unique<concrete_holder_<I>>(std::forward<I>(impl)))
    {
    }

    explicit operator bool() const noexcept
    {
        return hld_ != nullptr;
    }

    inline bool has_data() const
    {
        return hld_->has_data;
    }

    inline bool has_eval() const
    {
        return hld_->has_eval;
    }

    inline std::vector<std::string> keys() const
    {
        return hld_->keys();
    }

    inline bool contains(const std::string& key) const
    {
        return hld_->contains(key);
    }

    inline icm::const_json data() const
    {
        return hld_->data();
    }

    array operator[](const std::string& key) const;

    inline dict eval() const;

    inline std::string serialize() const
    {
        if (hld_->has_serialize) {
            return hld_->serialize();
        } else if (hld_->has_data) {
            return hld_->data().dump();
        }
        return eval().serialize();
    }

    inline bool operator==(const dict& other) const
    {
        return data() == other.data();
    }

    inline bool operator!=(const dict& other) const
    {
        return data() != other.data();
    }

private:
    std::shared_ptr<holder_> hld_;
};

} // namespace nd

#include "array.hpp"

namespace nd {

template <typename I>
array dict::concrete_holder_<I>::operator[](const std::string& key) const
{
    if constexpr (impl::has_subscript_member_operator_v<I>) {
        return impl_[key];
    } else {
        return array(adapt(impl_.data().at(key)));
    }
}

template <typename I>
dict dict::concrete_holder_<I>::eval() const
{
    if constexpr (impl::has_dict_eval_member_function_v<I>) {
        return impl_.eval();
    } else {
        throw invalid_operation("eval() method is not implemented for this array.");
    }
}

inline array dict::operator[](const std::string& key) const
{
    return hld_->operator[](key);
}

inline dict dict::eval() const
{
    if (hld_->has_data) {
        return *this;
    }
    return hld_->eval();
}

} // namespace nd
