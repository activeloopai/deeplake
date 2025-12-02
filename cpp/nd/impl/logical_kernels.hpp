#include "chained_iterator.hpp"

namespace nd::impl {

array create_not_kernel(array&& a);

array create_and_kernel(array&& a, array&& b);

array create_or_kernel(array&& a, array&& b);

class not_kernel_expression
{
public:
    not_kernel_expression(array&& a)
        : a_(std::move(a))
    {
    }

    not_kernel_expression(const not_kernel_expression&) = default;
    not_kernel_expression& operator=(const not_kernel_expression&) = delete;
    not_kernel_expression(not_kernel_expression&&) noexcept = default;
    not_kernel_expression& operator=(not_kernel_expression&&) noexcept = default;

    ~not_kernel_expression() = default;

    enum dtype dtype() const
    {
        return dtype::boolean;
    }

    icm::shape shape() const
    {
        return a_.shape();
    }

    bool value(int64_t index) const
    {
        return !a_.value<bool>(index);
    }

    using iterator = chained_iterator<nd::array (*)(nd::array&&), nd::iterator>;

    iterator begin() const
    {
        return iterator(create_not_kernel, a_.begin());
    }

    iterator end() const
    {
        return iterator(create_not_kernel, a_.end());
    }

    array get(int64_t index) const
    {
        return create_not_kernel(a_[index]);
    }

    array eval() const
    {
        auto a = ::nd::eval(a_);
        return switch_numeric_dtype(a.dtype(), [&a]<typename U>() {
            auto sp = a.template data<U>();
            icm::vector<bool> res;
            res.reserve(sp.size());
            for (auto v : sp) {
                res.push_back(!v);
            }
            return ::nd::adapt(std::move(res), a.shape());
        });
    }

    void copy_data(std::span<uint8_t> buffer) const
    {
        auto a = ::nd::eval(a_);
        return switch_numeric_dtype(a.dtype(), [&a, &buffer]<typename U>() {
            auto sp = a.template data<U>();
            auto res = base::span_cast<bool>(buffer);
            ASSERT(res.size() == sp.size());
            for (auto i = 0; i < sp.size(); ++i) {
                res[i] = !sp[i];
            }
        });
    }

    void nonzero(icm::bit_vector_view bv) const
    {
        nd::nonzero(a_, bv);
        bv.flip_all();
    }

    uint8_t dimensions() const
    {
        return a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return a_.is_dynamic();
    }

private:
    array a_;
};

template <bool is_and>
class binary_logical_kernel_expression
{
public:
    binary_logical_kernel_expression(array&& a, array&& b)
        : a_(std::move(a))
        , b_(std::move(b))
    {
    }

    binary_logical_kernel_expression(const binary_logical_kernel_expression&) = default;
    binary_logical_kernel_expression& operator=(const binary_logical_kernel_expression&) = delete;
    binary_logical_kernel_expression(binary_logical_kernel_expression&&) noexcept = default;
    binary_logical_kernel_expression& operator=(binary_logical_kernel_expression&&) noexcept = default;

    ~binary_logical_kernel_expression() = default;

    enum dtype dtype() const
    {
        return dtype::boolean;
    }

    icm::shape shape() const
    {
        return a_.shape();
    }

    bool value(int64_t index) const
    {
        if constexpr (is_and) {
            return a_.value<bool>(index) && b_.value<bool>(index);
        } else {
            return a_.value<bool>(index) || b_.value<bool>(index);
        }
    }

    using iterator = chained_iterator<nd::array (*)(nd::array&&, nd::array&&), nd::iterator, nd::iterator>;

    iterator begin() const
    {
        if constexpr (is_and) {
            return iterator(create_and_kernel, a_.begin(), b_.begin());
        } else {
            return iterator(create_or_kernel, a_.begin(), b_.begin());
        }
    }

    iterator end() const
    {
        if constexpr (is_and) {
            return iterator(create_and_kernel, a_.end(), b_.end());
        } else {
            return iterator(create_or_kernel, a_.end(), b_.end());
        }
    }

    array get(int64_t index) const
    {
        if constexpr (is_and) {
            return create_and_kernel(a_[index], b_[index]);
        } else {
            return create_or_kernel(a_[index], b_[index]);
        }
    }

    void nonzero(icm::bit_vector_view bv) const
    {
        nd::nonzero(a_, bv);
        icm::bit_vector bbv(b_.size());
        nd::nonzero(b_, bbv.span(0, b_.size()));
        if constexpr (is_and) {
            bv &= bbv.span(0, b_.size());
        } else {
            bv |= bbv.span(0, b_.size());
        }
    }

    uint8_t dimensions() const
    {
        return a_.dimensions();
    }

    constexpr bool is_dynamic() const noexcept
    {
        return a_.is_dynamic();
    }

private:
    array a_;
    array b_;
};

inline array create_not_kernel(array&& a)
{
    return array(not_kernel_expression(std::move(a)));
}

inline array create_and_kernel(array&& a, array&& b)
{
    return array(binary_logical_kernel_expression<true>(std::move(a), std::move(b)));
}

inline array create_or_kernel(array&& a, array&& b)
{
    return array(binary_logical_kernel_expression<false>(std::move(a), std::move(b)));
}

}
