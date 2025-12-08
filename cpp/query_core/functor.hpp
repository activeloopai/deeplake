#pragma once

#include "expr.hpp"
#include "static_data_t.hpp"

#include <deeplake_core/sample.hpp>
#include <deeplake_core/sample_range.hpp>

#include <base/htype.hpp>
#include <nd/adapt.hpp>
#include <nd/comparison.hpp>
#include <nd/functions.hpp>
#include <nd/type.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <functional>
#include <string>

namespace query_core {

template <typename R, typename T>
class functor_t
{
public:
    functor_t() = default;

    template <typename F>
    requires(std::is_invocable_v<F> && std::is_same_v<R, std::invoke_result_t<F>>)
    explicit functor_t(F&& f, expr e, const nd::type& t, base::htype ht)
        : function_([f = std::forward<F>(f)](const T&, const static_data_t&, int64_t) {
            return f();
        })
        , expr_(std::move(e))
        , type_(t)
        , htype_(ht)
        , is_const_(true)
    {
    }

    template <typename F>
    requires(std::is_invocable_v<F, static_data_t> && std::is_same_v<R, std::invoke_result_t<F, static_data_t>>)
    explicit functor_t(F&& f, expr e, const nd::type& t, base::htype ht)
        : function_([f = std::forward<F>(f)](const T&, const static_data_t& d, int64_t) {
            return f(d);
        })
        , expr_(std::move(e))
        , type_(t)
        , htype_(ht)
        , is_const_(true)
    {
    }

    template <typename F>
    requires(std::is_invocable_v<F, T> && std::is_same_v<R, std::invoke_result_t<F, T>>)
    explicit functor_t(F&& f, expr e, const nd::type& t, base::htype ht, bool is_identity = false)
        : function_([f = std::forward<F>(f)](const T& s, const static_data_t&, int64_t) {
              return f(s);
          }),
          expr_(std::move(e)), type_(t), htype_(ht), is_identity_(is_identity)
    {
    }

    template <typename F>
    requires(std::is_invocable_v<F, T, static_data_t> && std::is_same_v<R, std::invoke_result_t<F, T, static_data_t>>)
    explicit functor_t(F&& f, expr e, const nd::type& t, base::htype ht, bool is_identity = false)
        : function_([f = std::forward<F>(f)](const T& s, const static_data_t& d, int64_t) {
              return f(s, d);
          }), expr_(std::move(e)), type_(t), htype_(ht), is_identity_(is_identity)
    {
    }

    template <typename F>
    requires(std::is_invocable_v<F, T, static_data_t, int64_t> &&
             std::is_same_v<R, std::invoke_result_t<F, T, static_data_t, int64_t>>)
    explicit functor_t(F&& f, expr e, const nd::type& t, base::htype ht, bool is_identity = false)
        : function_(std::forward<F>(f))
        , expr_(std::move(e))
        , type_(t)
        , htype_(ht)
        , is_identity_(is_identity)
    {
    }

public:
    explicit operator bool() const
    {
        return static_cast<bool>(function_);
    }

    inline const expr& get_expr() const
    {
        return expr_;
    }

    inline void set_expr(expr e)
    {
        expr_ = std::move(e);
    }

    inline bool is_const() const
    {
        return is_const_;
    }

    inline bool is_identity() const
    {
        return is_identity_;
    }

    inline nd::type type() const
    {
        return type_;
    }

    inline nd::dtype dtype() const
    {
        return type_.get_dtype();
    }

    inline base::htype htype() const
    {
        return htype_;
    }

    inline R operator()(const T& s, const static_data_t& d, int64_t param_index) const
    {
        return function_(s, d, param_index);
    }

private:
    std::function<R(const T&, const static_data_t&, int64_t)> function_;
    expr expr_;
    nd::type type_ = nd::type::unknown();
    base::htype htype_;
    bool is_identity_ = false;
    bool is_const_ = false;
};

template <typename R>
using sample_functor = functor_t<R, const deeplake_core::sample&>;
template <typename R>
using batch_functor = functor_t<R, deeplake_core::sample_range>;

template <typename T>
struct batch_functor_return_type
{
    using type = icm::vector<T>;

    static inline type convert(icm::vector<T>&& v)
    {
        return std::move(v);
    }

    static inline type repeat(const T& v, int64_t n)
    {
        type ret;
        ret.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            ret.push_back(v);
        }
        return ret;
    }
};

template <>
struct batch_functor_return_type<nd::array>
{
    using type = nd::array;

    static inline type convert(icm::vector<nd::array>&& v)
    {
        return nd::dynamic(std::move(v));
    }

    static inline type repeat(nd::array v, int64_t n)
    {
        return nd::dynamic_repeated(std::move(v), n);
    }
};

template <typename T>
using batch_functor_return_type_t = typename batch_functor_return_type<T>::type;

using array_sample_functor = sample_functor<nd::array>;
using json_sample_functor = sample_functor<icm::const_json>;
using array_batch_functor = batch_functor<batch_functor_return_type_t<nd::array>>;
using json_batch_functor = batch_functor<batch_functor_return_type_t<icm::const_json>>;

template <typename R>
class parametric_const_functor
{
public:
    parametric_const_functor() = default;

    parametric_const_functor(std::vector<R> values, expr e, nd::type t, base::htype ht)
        : values_(std::move(values)), expr_(std::move(e)), type_(std::move(t)), htype_(ht)
    {
    }

    parametric_const_functor(const parametric_const_functor&) noexcept = default;
    parametric_const_functor(parametric_const_functor&&) noexcept = default;
    parametric_const_functor& operator=(const parametric_const_functor&) noexcept = default;
    parametric_const_functor& operator=(parametric_const_functor&&) noexcept = default;
    ~parametric_const_functor() noexcept = default;

    inline R operator()(int64_t param_index) const
    {
        ASSERT(param_index >= 0 && param_index < static_cast<int64_t>(values_.size()));
        return values_[param_index];
    }

    explicit operator bool() const
    {
        return static_cast<bool>(values_);
    }

    inline const expr& get_expr() const
    {
        return expr_;
    }

    inline void set_expr(expr e)
    {
        expr_ = std::move(e);
    }

    inline nd::dtype dtype() const
    {
        return type_.get_dtype();
    }

    inline const nd::type& type() const
    {
        return type_;
    }

    inline const base::htype& htype() const
    {
        return htype_;
    }

    inline bool is_const() const
    {
        return true;
    }

    inline bool is_identity() const
    {
        return false;
    }

private:
    std::vector<R> values_;
    expr expr_;
    nd::type type_;
    base::htype htype_;
};

template <typename R>
class generic_functor
{
public:
    generic_functor() = default;

    generic_functor(sample_functor<R>&& f) : f_(std::move(f))
    {
    }

    generic_functor(batch_functor<batch_functor_return_type_t<R>>&& f) : f_(std::move(f))
    {
    }

    generic_functor(parametric_const_functor<R>&& f) : f_(std::move(f))
    {
    }

public:
    inline explicit operator bool() const
    {
        return !is_sample() || static_cast<bool>(std::get<0>(f_));
    }

    inline bool is_sample() const
    {
        return f_.index() == 0;
    }

    inline bool is_batch() const
    {
        return f_.index() == 1;
    }

    inline bool is_parametric() const
    {
        return f_.index() == 2;
    }

    inline const expr& get_expr() const
    {
        return std::visit([](const auto& f) -> const expr& {
            return f.get_expr();
        }, f_);
    }

    inline void set_expr(expr e)
    {
        return std::visit([&e](auto& f) {
            f.set_expr(std::move(e));
        }, f_);
    }

    inline bool is_const() const
    {
        return std::visit([](const auto& f) {
            return f.is_const();
        }, f_);
    }

    inline bool is_identity() const
    {
        return std::visit([](const auto& f) {
            return f.is_identity();
        }, f_);
    }

    inline nd::dtype dtype() const
    {
        return std::visit([](const auto& f) {
            return f.dtype();
        }, f_);
    }

    inline nd::type type() const
    {
        return std::visit([](const auto& f) {
            return f.type();
        }, f_);
    }

    inline base::htype htype() const
    {
        return std::visit([](const auto& f) {
            return f.htype();
        }, f_);
    }

    inline batch_functor_return_type_t<R> operator()(const deeplake_core::sample_range& b,
                                                     const static_data_t& d, int64_t param_index) const
    {
        return batch_f()(b, d, param_index);
    }

    batch_functor<batch_functor_return_type_t<R>> batch_f() const
    {
        if (is_sample()) {
            auto f = std::get<0>(f_);
            return batch_functor<batch_functor_return_type_t<R>>(
                [f](const deeplake_core::sample_range& b, const static_data_t& d, int64_t param_index) {
                    icm::vector<R> ret;
                    ret.reserve(b.size());
                    for (auto s : b) {
                        ret.emplace_back(f(s, d, param_index));
                    }
                    return batch_functor_return_type<R>::convert(std::move(ret));
                },
                f.get_expr(),
                f.type(),
                f.htype(),
                f.is_identity());
        } else if (is_batch()) {
            return std::get<1>(f_);
        }
        auto f = std::get<2>(f_);
        return batch_functor<batch_functor_return_type_t<R>>(
            [f = std::move(f)](const deeplake_core::sample_range& r, const static_data_t&, int64_t param_index) {
                auto s = r.size();
                auto v = f(param_index);
                return batch_functor_return_type<R>::repeat(std::move(v), s);
            },
            f.get_expr(), f.type(), f.htype(), f.is_identity());
    }

private:
    std::variant<sample_functor<R>, batch_functor<batch_functor_return_type_t<R>>, parametric_const_functor<R>> f_;
};

using array_generic_functor = generic_functor<nd::array>;
using json_generic_functor = generic_functor<icm::const_json>;
using array_parametric_const_functor = parametric_const_functor<nd::array>;

} // namespace query_core





