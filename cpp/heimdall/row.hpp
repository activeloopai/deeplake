#pragma once

#include "row_view.hpp"

namespace heimdall {

class dataset;
class row;

class row : public row_view
{
public:
    row(std::shared_ptr<dataset> view, int64_t row_id);

    [[nodiscard]] std::shared_ptr<dataset> mutable_view() const;

    [[nodiscard]] async::promise<void> set_value(const std::string& column_name, const nd::array& value);
};

namespace impl {

row row_constructor(dataset& ds, int64_t index);

class row_based_iterator
{
public:
    row_based_iterator(dataset& view,
                       icm::index_mapping_t<int64_t>::iterator index_iterator,
                       row (*constructor)(dataset&, int64_t))
        : view_(&view)
        , index_iterator_(index_iterator)
        , constructor_(constructor)
    {
    }

    row_based_iterator(dataset& view, int64_t index, row (*constructor)(dataset&, int64_t))
        : view_(&view)
        , index_iterator_(index)
        , constructor_(constructor)
    {
    }

    row_based_iterator& operator++()
    {
        std::visit(
            [](auto& iterator) mutable {
                ++iterator;
            },
            index_iterator_);
        return *this;
    }

    row_based_iterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    row_based_iterator& operator--()
    {
        std::visit(
            [](auto& iterator) mutable {
                --iterator;
            },
            index_iterator_);
        return *this;
    }

    row_based_iterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    row_based_iterator& operator+=(int64_t value)
    {
        std::visit(
            [value](auto& iterator) mutable {
                iterator += value;
            },
            index_iterator_);
        return *this;
    }

    row_based_iterator operator+(int64_t value) const
    {
        auto copy = *this;
        copy += value;
        return copy;
    }

    row_based_iterator& operator-=(int64_t value)
    {
        std::visit(
            [value](auto& iterator) mutable {
                iterator -= value;
            },
            index_iterator_);
        return *this;
    }

    row_based_iterator operator-(int64_t value) const
    {
        auto copy = *this;
        copy -= value;
        return copy;
    }

    int64_t operator-(const row_based_iterator& value) const
    {
        return std::visit(
            [](const auto& i1, const auto& i2) {
                if constexpr (std::is_same_v<decltype(i1), decltype(i2)>) {
                    return i1 - i2;
                } else {
                    ASSERT(false);
                    return static_cast<int64_t>(0L);
                }
            },
            index_iterator_,
            value.index_iterator_);
    }

    row operator*() const
    {
        if (std::holds_alternative<int64_t>(index_iterator_)) {
            return constructor_(*view_, std::get<int64_t>(index_iterator_));
        }
        return constructor_(*view_, *std::get<icm::index_mapping_t<int64_t>::iterator>(index_iterator_));
    }

    bool operator==(const row_based_iterator& value) const
    {
        return view_ == value.view_ && index_iterator_ == value.index_iterator_;
    }

private:
    dataset* view_;
    std::variant<int64_t, icm::index_mapping_t<int64_t>::iterator> index_iterator_;
    row (*constructor_)(dataset&, int64_t);
};

}
}
