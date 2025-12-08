#pragma once

#include <async/promise.hpp>
#include <base/format.hpp>
#include <icm/index_mapping.hpp>
#include <nd/array.hpp>
#include <nd/type.hpp>

namespace heimdall {

class row_view;
class dataset_view;

class row_view
{
public:
    row_view(std::shared_ptr<dataset_view> view, int64_t row_id)
        : view_(std::move(view))
        , row_id_(row_id)
    {
    }

    bool operator==(const row_view& value) const
    {
        return row_id_ == value.row_id_ && view_ == value.view_;
    }

    [[nodiscard]] int64_t row_id() const
    {
        return row_id_;
    }

    [[nodiscard]] const auto& view() const
    {
        return view_;
    }

    [[nodiscard]] async::promise<nd::array> value(const std::string& column_name) const;

    [[nodiscard]] async::promise<nd::array> bytes(const std::string& column_name) const;

    [[nodiscard]] std::string to_string() const
    {
        return fmt::format("Row({})", row_id_);
    }

private:
    std::shared_ptr<dataset_view> view_;
    int64_t row_id_;
};

namespace impl {

row_view row_view_constructor(dataset_view& ds, int64_t index);

class row_view_based_iterator
{
public:
    row_view_based_iterator(dataset_view& view,
                            icm::index_mapping_t<int64_t>::iterator index_iterator,
                            row_view (*constructor)(dataset_view&, int64_t))
        : view_(&view)
        , index_iterator_(index_iterator)
        , constructor_(constructor)
    {
    }

    row_view_based_iterator(dataset_view& view, int64_t index, row_view (*constructor)(dataset_view&, int64_t))
        : view_(&view)
        , index_iterator_(index)
        , constructor_(constructor)
    {
    }

    row_view_based_iterator& operator++()
    {
        std::visit(
            [](auto& iterator) mutable {
                ++iterator;
            },
            index_iterator_);
        return *this;
    }

    row_view_based_iterator operator++(int)
    {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    row_view_based_iterator& operator--()
    {
        std::visit(
            [](auto& iterator) mutable {
                --iterator;
            },
            index_iterator_);
        return *this;
    }

    row_view_based_iterator operator--(int)
    {
        auto copy = *this;
        --(*this);
        return copy;
    }

    row_view_based_iterator& operator+=(int64_t value)
    {
        std::visit(
            [value](auto& iterator) mutable {
                iterator += value;
            },
            index_iterator_);
        return *this;
    }

    row_view_based_iterator operator+(int64_t value) const
    {
        auto copy = *this;
        copy += value;
        return copy;
    }

    row_view_based_iterator& operator-=(int64_t value)
    {
        std::visit(
            [value](auto& iterator) mutable {
                iterator -= value;
            },
            index_iterator_);
        return *this;
    }

    row_view_based_iterator operator-(int64_t value) const
    {
        auto copy = *this;
        copy -= value;
        return copy;
    }

    int64_t operator-(const row_view_based_iterator& value) const
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

    row_view operator*() const
    {
        if (std::holds_alternative<int64_t>(index_iterator_)) {
            return constructor_(*view_, std::get<int64_t>(index_iterator_));
        }
        return constructor_(*view_, *std::get<icm::index_mapping_t<int64_t>::iterator>(index_iterator_));
    }

    bool operator==(const row_view_based_iterator& value) const
    {
        return view_ == value.view_ && index_iterator_ == value.index_iterator_;
    }

private:
    dataset_view* view_;
    std::variant<int64_t, icm::index_mapping_t<int64_t>::iterator> index_iterator_;
    row_view (*constructor_)(dataset_view&, int64_t);
};

}

} // namespace heimdall
