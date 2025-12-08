#pragma once

#include "exceptions.hpp"

#include <query_core/functor.hpp>
#include <icm/const_json.hpp>
#include <nd/type.hpp>

#include <variant>

namespace query_core {

class order_functor
{
public:
    order_functor() = default;

    explicit order_functor(query_core::generic_functor<int> f) : func_(std::move(f))
    {
    }

    explicit order_functor(query_core::generic_functor<float> f) : func_(std::move(f))
    {
    }

    explicit order_functor(query_core::generic_functor<std::string> f) : func_(std::move(f))
    {
    }

    explicit order_functor(query_core::json_generic_functor f) : func_(std::move(f))
    {
    }

public:
    inline bool has_func() const
    {
        switch (func_.index()) {
        case 0:
            return static_cast<bool>(std::get<query_core::generic_functor<int>>(func_));
        case 1:
            return static_cast<bool>(std::get<query_core::generic_functor<float>>(func_));
        case 2:
            return static_cast<bool>(std::get<query_core::generic_functor<std::string>>(func_));
        case 3:
            return static_cast<bool>(std::get<query_core::generic_functor<icm::const_json>>(func_));
        default:
            ASSERT("Invalid variant");
        }
        return false;
    }

    template <typename F>
    inline auto switch_func(F f) const
    {
        switch (func_.index()) {
        case 0:
            return f.template operator()<int>(std::get<query_core::generic_functor<int>>(func_));
        case 1:
            return f.template operator()<float>(std::get<query_core::generic_functor<float>>(func_));
        case 2:
            return f.template operator()<std::string>(std::get<query_core::generic_functor<std::string>>(func_));
        case 3:
            return f.template operator()<icm::const_json>(
                std::get<query_core::generic_functor<icm::const_json>>(func_));
        default:
            ASSERT("Invalid variant");
        }
        throw invalid_order_function();
    }

    inline const query_core::expr& get_expr() const
    {
        return std::visit(
            [](const auto& arg) -> const query_core::expr& {
                return arg.get_expr();
            },
            func_);
    }

private:
    std::variant<query_core::generic_functor<int>, query_core::generic_functor<float>,
                 query_core::generic_functor<std::string>, query_core::generic_functor<icm::const_json>>
        func_;
};

template <typename T>
inline query_core::generic_functor<T> meta_to_generic(const query_core::array_generic_functor& f)
{
    auto t = f.type();

    auto l = [g = f](const deeplake_core::sample_range& s, const query_core::static_data_t& d, int64_t param_index) {
        auto vs = g(s, d, param_index);
        icm::vector<T> ret(vs.size());
        for (auto i = 0; i < vs.size(); ++i) {
            auto v = vs[i];
            if (v.volume() != 1) {
                throw exception("Can't convert array to scalar in ORDER/SAMPLE/GROUP expressions.");
            }
            ret[i] = v.value<T>(0);
        }
        return ret;
    };
    auto bf = query_core::batch_functor<icm::vector<T>>(
        std::move(l), f.get_expr(), nd::type::cast(t, nd::dtype::float32), base::htype::generic);
    return query_core::generic_functor<T>(std::move(bf));
}

inline order_functor meta_to_order(const query_core::array_generic_functor& f)
{
    if (f.dtype() == nd::dtype::object) {
        auto l = [g = std::move(f)](
                     const deeplake_core::sample_range& s, const query_core::static_data_t& d, int64_t param_index) {
            auto vs = g(s, d, param_index);
            icm::vector<icm::const_json> ret;
            ret.reserve(vs.size());
            for (auto i = 0; i < vs.size(); ++i) {
                auto v = vs[i];
                if (v.is_dynamic()) {
                    throw exception("Can't convert array to json in ORDER expression.");
                }
                nd::switch_dtype(v.dtype(), [&ret, &v, i, g]<typename T>() {
                    if constexpr (std::is_same_v<T, nd::dict>) {
                        if (g.htype() == base::htype::json && v.shape().empty()) {
                            ret.emplace_back(icm::const_json::null_json());
                        } else {
                            ret.emplace_back(v.dict_value(0).data());
                        }
                    } else if constexpr (std::is_same_v<T, nd::dtype_type_t<nd::dtype::string>>) {
                        ret.emplace_back(icm::const_json(base::string_view_cast(v.template data<char>())));
                    } else if constexpr (std::is_same_v<T, nd::dtype_type_t<nd::dtype::byte>>) {
                        ret.emplace_back(icm::const_json());
                    } else {
                        ret.emplace_back(icm::const_json(v.value<T>(0)));
                    }
                });
            }
            return ret;
        };
        auto bf = query_core::batch_functor<icm::vector<icm::const_json>>(std::move(l), f.get_expr(), f.type(),
                                                                                base::htype::json);
        return order_functor(query_core::generic_functor<icm::const_json>(std::move(bf)));
    }
    if (f.dtype() == nd::dtype::string) {
        auto l = [g = std::move(f)](
                     const deeplake_core::sample_range& s, const query_core::static_data_t& d, int64_t param_index) {
            auto vs = g(s, d, param_index);
            icm::vector<std::string> ret(vs.size());
            for (auto i = 0; i < vs.size(); ++i) {
                auto v = vs[i];
                if (v.is_dynamic()) {
                    throw exception("Can't convert list of strings to string in ORDER expression.");
                }
                ret[i] = base::string_view_cast(v.template data<const char>());
            }
            return ret;
        };
        auto bf = query_core::batch_functor<icm::vector<std::string>>(std::move(l), f.get_expr(), f.type(),
                                                                         base::htype::text);
        return order_functor(query_core::generic_functor<std::string>(std::move(bf)));
    } else {
        return order_functor(meta_to_generic<float>(f));
    }
}

} // namespace query_core
