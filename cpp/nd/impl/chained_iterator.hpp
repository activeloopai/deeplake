#pragma once

namespace nd::impl {

template <typename F, typename ... Its>
class chained_iterator
{
public:
    chained_iterator(F f, Its&&... its)
        : its_(std::move(its)...)
        , f_(std::move(f))
    {
    }

    array operator*() const
    {
        return std::apply(
            [this](auto&... iterators) {
                return f_(*iterators...);
            },
            its_);
    }

    void operator++()
    {
        apply([](auto& it) {
            ++it;
        });
    }

    void operator--()
    {
        apply([](auto& it) {
            --it;
        });
    }

    void operator+=(std::ptrdiff_t n)
    {
        apply([n](auto& it) {
            it += n;
        });
    }

    void operator-=(std::ptrdiff_t n)
    {
        apply([n](auto& it) {
            it -= n;
        });
    }

    std::ptrdiff_t operator-(const chained_iterator& other) const
    {
        return std::get<0>(its_) - std::get<0>(other.its_);
    }

    bool operator==(const chained_iterator& other) const
    {
        return its_ == other.its_;
    }

private:
    template <typename Func>
    inline void apply(Func&& func)
    {
        std::apply(
            [&func](auto&... iterators) {
                (func(iterators), ...);
            },
            its_);
    }

private:
    std::tuple<Its ...> its_;
    F f_;
};


}
