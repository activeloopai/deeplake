#pragma once

#include "../handle_base.hpp"

namespace async::impl {

template <typename Result>
class fulfilled_promise : public handle_base<Result>
{
    using base_t = handle_base<Result>;

public:
    explicit fulfilled_promise(Result v = Result())
        : base_t(result<Result>(std::move(v)))
    {
    }
};

template <>
class fulfilled_promise<void> : public handle_base<void>
{
    using base_t = handle_base<void>;
public:
    explicit fulfilled_promise()
        : base_t(result<void>(true))
    {
    }
};

template <typename Result>
class error_promise : public handle_base<Result>
{
    using base_t = handle_base<Result>;

public:
    explicit error_promise(std::exception_ptr e)
        : base_t(std::move(e))
    {
    }
};

}
