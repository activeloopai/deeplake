#pragma once

#include "../handle_base.hpp"

#include <numeric>
#include <tuple>

namespace async::impl {

template <typename T>
struct multiple_promises_state_t
{
    using results_t = std::vector<T>;

    explicit multiple_promises_state_t(std::vector<promise<T>>&& promises)
        : promises_(std::move(promises))
        , results_(promises_.size())
        , completed_count_(0)
    {
    }

    multiple_promises_state_t(multiple_promises_state_t&& other) noexcept
        : promises_(std::move(other.promises_))
        , results_(std::move(other.results_))
        , completed_count_(other.completed_count_.load())
    {
    }

    std::vector<promise<T>> promises_;
    results_t results_;
    std::atomic<std::size_t> completed_count_;
};

template<>
struct multiple_promises_state_t<void>
{
    using results_t = void;

    explicit multiple_promises_state_t(std::vector<promise<void>>&& promises)
        : promises_(std::move(promises))
        , completed_count_(0)
    {
    }

    multiple_promises_state_t(multiple_promises_state_t&& other) noexcept
        : promises_(std::move(other.promises_))
        , completed_count_(other.completed_count_.load())
    {
    }

    std::vector<promise<void>> promises_;
    std::atomic<std::size_t> completed_count_;
};


template <typename T>
class multiple_promises
    : public handle_base<typename multiple_promises_state_t<T>::results_t, multiple_promises_state_t<T>>
{
    using state_t = multiple_promises_state_t<T>;
    using base_t = handle_base<typename state_t::results_t, state_t>;

    using typename base_t::data_type;

public:
    explicit multiple_promises(std::vector<promise<T>>&& p_list)
        : base_t(state_t(std::move(p_list)))
    {
        auto& ps = promises();
        for (std::size_t i = 0; i < ps.size(); ++i) {
            ps[i].set_callback({[*this, i](result<T>&& r) mutable {
                                    if (base_t::is_unresolved()) {
                                        try {
                                            set_result_(i, std::move(r));
                                        } catch (...) {
                                            base_t::set_exception(std::current_exception());
                                        }
                                    }
                                },
                                nullptr});
        }
    }

    multiple_promises(const multiple_promises&) = default;
    multiple_promises& operator=(const multiple_promises&) = delete;
    multiple_promises(multiple_promises&&) noexcept = default;
    multiple_promises& operator=(multiple_promises&&) = delete;
    ~multiple_promises() = default;

    void set_priority(int priority)
    {
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        if (base_t::is_cancelled(data)) {
            return;
        }
        for (auto& p : promises()) {
            p.set_priority(priority);
        }
    }

    bool cancel()
    {
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        if (base_t::is_cancelled(data)) {
            return true;
        }
        if (!base_t::cancel(data)) {
            return false;
        }
        for (auto& p : promises()) {
            p.cancel();
        }
        promises().clear();
        return true;
    }

    float progress() const
    {
        const auto& data = base_t::data();
        std::scoped_lock lock(data->lock_);
        const auto& ps = promises();
        return std::accumulate(ps.begin(), ps.end(), 0.0f,
                               [](const auto& f, const auto& p) {
                                   return f + p.progress();
                               }) /
               ps.size();
    }

private:
    inline void set_result_(std::size_t id, result<T>&& r)
    {
        if constexpr (std::is_void_v<T>) {
            r.get();
        } else {
            ASSERT(id < base_t::state().results_.size());
            base_t::state().results_[id] = std::move(r.get());
        }
        const auto completed_count = ++base_t::state().completed_count_;
        ASSERT(completed_count <= promises().size());
        if (completed_count == promises().size()) {
            if constexpr (std::is_void_v<T>) {
                base_t::set_value();
            } else {
                base_t::set_value(std::move(base_t::state().results_));
            }
        }
    }

    inline const std::vector<async::promise<T>>& promises() const noexcept
    {
        return base_t::state().promises_;
    }

    inline std::vector<async::promise<T>>& promises() noexcept
    {
        return base_t::state().promises_;
    }
};

} // namespace async::impl
