#pragma once

/**
 * @file array.hpp
 * @brief Definition and implementation of `array` class.
 */

#include "dtype.hpp"
#include "exceptions.hpp"
#include "impl/mpl.hpp"
#include "impl/range_array.hpp"
#include "impl/std_span_array.hpp"
#include "iterator.hpp"

#include <base/assert.hpp>
#include <base/bit_cast.hpp>
#include <base/span_cast.hpp>
#include <icm/bit_vector.hpp>
#include <icm/index_mapping.hpp>
#include <icm/shape.hpp>
#include <icm/slice.hpp>
#include <icm/small_vector.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace nd {

class dict;

array eval(array arr);
void copy_data(const array& arr, std::span<uint8_t> buffer);

template <typename I>
array stride(array arr, icm::slice_t<I> slice);

void nonzero(const array& arr, icm::bit_vector_view output);

template <typename T>
const T* dynamic_array_cast(const array& arr);

template <typename T>
const T* static_array_cast(const array& arr);

constexpr std::size_t s_array_bytes = 39;

class array
{
private:
    enum class mode_t : uint8_t
    {
        null,
        scalar_inplace,
        holder_inplace,
        holder_pointer
    };

    using void_ptr = void*;

    template <std::size_t s>
    using byte_t = std::array<char, s>;

    struct holder_
    {

        virtual void copy_to(void_ptr) const = 0;
        virtual void move_to(void_ptr) = 0;

        virtual ~holder_() = default;

        virtual icm::shape shape() const = 0;
        virtual std::span<const uint8_t> data() const = 0;

        virtual byte_t<1> byte_1_value(int64_t) const = 0;
        virtual byte_t<2> byte_2_value(int64_t) const = 0;
        virtual byte_t<4> byte_4_value(int64_t) const = 0;
        virtual byte_t<8> byte_8_value(int64_t) const = 0;
        virtual dict dict_value(int64_t) const = 0;

        virtual iterator begin() const = 0;
        virtual iterator end() const = 0;
        virtual array get(int64_t) const = 0;
        virtual array get_range(int64_t, int64_t) const = 0;
        virtual array eval() const = 0;
        virtual void copy_data(std::span<uint8_t>) const = 0;

        virtual array stride(icm::index_mapping_t<int64_t>&& source_shape) const = 0;
        virtual array stride(icm::index_mapping_t<int>&& source_shape) const = 0;
        virtual void nonzero(icm::bit_vector_view) const = 0;
    };
    using holder_ptr = std::shared_ptr<holder_>;

    template <typename I>
    struct concrete_holder_ final : public holder_
    {
        static_assert(impl::has_data_member_function_v<I> || impl::has_get_member_function_v<I>,
                      "Array adaptors should implement at least one of the following functions:\n"
                      "\tstd::span<const uint8_t> data() const;\n"
                      "\tnd::array get(int64_t index) const;\n");

        static constexpr int value_funcs_count = static_cast<int>(impl::has_value_member_function_v<I, bool>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, char>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, uint8_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, int8_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, uint16_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, int16_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, uint32_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, int32_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, uint64_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, int64_t>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, float>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, double>) +
                                                 static_cast<int>(impl::has_value_member_function_v<I, dict>);

        static_assert(value_funcs_count <= 1,
                      "Array adaptors should implement no more than one of the value functions.");

        static_assert(value_funcs_count == 1 || impl::has_data_member_function_v<I> ||
                          impl::has_get_member_function_v<I>,
                      "Non-dynamic array adaptors should implement `data()`, `value()` or `get()` member functions.");

        explicit concrete_holder_(I&& i)
            : impl_(std::move(i))
        {
        }

        explicit concrete_holder_(const I& i)
            : impl_(i)
        {
        }

        void copy_to(void_ptr p) const override
        {
            if constexpr (std::is_copy_constructible_v<I>) {
                new (p) concrete_holder_(impl_);
            } else {
                ASSERT_MESSAGE(false, "Array implementation is not copy constructable.");
            }
        }

        void move_to(void_ptr p) override
        {
            new (p) concrete_holder_(std::move(impl_));
        }

        ~concrete_holder_() override = default;

        icm::shape shape() const override
        {
            return impl_.shape();
        }

        std::span<const uint8_t> data() const override
        {
            if constexpr (impl::has_data_member_function_v<I>) {
                return impl_.data();
            } else {
                throw invalid_operation("data() method is not implemented for this array.");
            }
        }

        byte_t<1> byte_1_value(int64_t index) const override
        {
            if constexpr (impl::has_value_member_function_v<I, bool> || impl::has_value_member_function_v<I, uint8_t> ||
                          impl::has_value_member_function_v<I, int8_t>) {
                ASSERT(impl_.dtype() == dtype::boolean || impl_.dtype() == dtype::uint8 ||
                       impl_.dtype() == dtype::int8 || impl_.dtype() == dtype::object);
                return base::bit_cast<byte_t<1>>(impl_.value(index));
            } else if constexpr (impl::has_data_member_function_v<I>) {
                return base::span_cast<const byte_t<1>>(impl_.data())[index];
            } else if constexpr (impl::has_get_member_function_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>());
                return base::bit_cast<byte_t<1>>(get(index / subvolume).template value<uint8_t>(index % subvolume));
            } else {
                throw invalid_operation("Can't convert array value to 1-byte.");
                return byte_t<1>();
            }
        }

        byte_t<2> byte_2_value(int64_t index) const override
        {
            if constexpr (impl::has_value_member_function_v<I, uint16_t> ||
                          impl::has_value_member_function_v<I, int16_t>) {
                ASSERT(impl_.dtype() == dtype::uint16 || impl_.dtype() == dtype::int16);
                return base::bit_cast<byte_t<2>>(impl_.value(index));
            } else if constexpr (impl::has_data_member_function_v<I>) {
                return base::span_cast<const byte_t<2>>(impl_.data())[index];
            } else if constexpr (impl::has_get_member_function_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>());
                return base::bit_cast<byte_t<2>>(get(index / subvolume).template value<uint16_t>(index % subvolume));
            } else {
                throw invalid_operation("Can't convert array value to 2-byte.");
                return byte_t<2>();
            }
        }

        byte_t<4> byte_4_value(int64_t index) const override
        {
            if constexpr (impl::has_value_member_function_v<I, uint32_t> ||
                          impl::has_value_member_function_v<I, int32_t> ||
                          impl::has_value_member_function_v<I, float>) {
                ASSERT(impl_.dtype() == dtype::uint32 || impl_.dtype() == dtype::int32 ||
                       impl_.dtype() == dtype::float32);
                return base::bit_cast<byte_t<4>>(impl_.value(index));
            } else if constexpr (impl::has_data_member_function_v<I>) {
                return base::span_cast<const byte_t<4>>(impl_.data())[index];
            } else if constexpr (impl::has_get_member_function_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>());
                return base::bit_cast<byte_t<4>>(get(index / subvolume).template value<uint32_t>(index % subvolume));
            } else {
                throw invalid_operation("Can't convert array value to 4-byte.");
                return byte_t<4>();
            }
        }

        byte_t<8> byte_8_value(int64_t index) const override
        {
            if constexpr (impl::has_value_member_function_v<I, uint64_t> ||
                          impl::has_value_member_function_v<I, int64_t> ||
                          impl::has_value_member_function_v<I, double>) {
                ASSERT(impl_.dtype() == dtype::uint64 || impl_.dtype() == dtype::int64 ||
                       impl_.dtype() == dtype::float64);
                return base::bit_cast<byte_t<8>>(impl_.value(index));
            } else if constexpr (impl::has_data_member_function_v<I>) {
                return base::span_cast<const byte_t<8>>(impl_.data())[index];
            } else if constexpr (impl::has_get_member_function_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>());
                return base::bit_cast<byte_t<8>>(get(index / subvolume).template value<uint64_t>(index % subvolume));
            } else {
                throw invalid_operation("Can't convert array value to 8-byte.");
                return byte_t<8>();
            }
        }

        dict dict_value(int64_t index) const override;

        iterator begin() const override
        {
            if constexpr (impl::has_data_member_function_v<I> && !impl::is_scalar_v<I> &&
                          !impl::has_is_none_member_variable_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>()) *
                                 nd::dtype_bytes(impl_.dtype());
                auto d = std::span<const uint8_t>(impl_.data().data(), subvolume);
                return iterator(data_iterator(impl_.owner(), d, icm::shape(sh.begin() + 1, sh.end()), impl_.dtype()));
            } else if constexpr (impl::has_begin_member_function_v<I>) {
                return iterator(impl_.begin());
            } else {
                throw invalid_operation("begin() method is not implemented for this array.");
            }
        }

        iterator end() const override
        {
            if constexpr (impl::has_data_member_function_v<I> && !impl::is_scalar_v<I> &&
                          !impl::has_is_none_member_variable_v<I>) {
                auto sh = shape();
                auto subvolume = std::accumulate(sh.begin() + 1, sh.end(), 1L, std::multiplies<int64_t>()) *
                                 nd::dtype_bytes(impl_.dtype());
                auto d = std::span<const uint8_t>(impl_.data().data() + impl_.data().size(), subvolume);
                return iterator(data_iterator(impl_.owner(), d, icm::shape(sh.begin() + 1, sh.end()), impl_.dtype()));
            } else if constexpr (impl::has_end_member_function_v<I>) {
                return iterator(impl_.end());
            } else {
                throw invalid_operation("end() method is not implemented for this array.");
            }
        }

        array get(int64_t index) const override;

        array get_range(int64_t start, int64_t end) const override;

        nd::array eval() const override
        {
            if constexpr (impl::has_eval_member_function_v<I>) {
                return impl_.eval();
            } else {
                throw invalid_operation("eval() method is not implemented for this array.");
            }
        }

        void copy_data(std::span<uint8_t> buffer) const override
        {
            if constexpr (impl::has_copy_data_member_function_v<I>) {
                return impl_.copy_data(buffer);
            } else {
                throw invalid_operation("copy_data(std::span<uint8_t>) method is not implemented for this array.");
            }
        }

        array stride(icm::index_mapping_t<int64_t>&& source_shape) const override
        {
            return stride_impl(std::move(source_shape));
        }

        array stride(icm::index_mapping_t<int>&& source_shape) const override
        {
            return stride_impl(std::move(source_shape));
        }

        void nonzero(icm::bit_vector_view output) const override
        {
            if constexpr (impl::has_nonzero_member_function_v<I>) {
                impl_.nonzero(output);
            } else {
                throw invalid_operation("nonzero(icm::bit_vector_view) method is not implemented for this array.");
            }
        }

    public:
        I impl_;

    private:
        template <typename V>
        array stride_impl(icm::index_mapping_t<V>&& source_shape) const
        {
            if constexpr (impl::has_template_stride_member_function_v<I>) {
                return impl_.stride(std::move(source_shape));
            } else {
                throw invalid_operation("stride(icm::index_mapping_t<V>&&) method is not implemented for this array.");
            }
        }
    };

    inline byte_t<1> byte_1_value(int64_t index) const
    {
        if (mode_ == mode_t::scalar_inplace) {
            ASSERT(index == 0);
            return base::span_cast<const byte_t<1>>(data())[index];
        }
        return holder()->byte_1_value(index);
    }

    inline byte_t<2> byte_2_value(int64_t index) const
    {
        if (mode_ == mode_t::scalar_inplace) {
            ASSERT(index == 0);
            return base::span_cast<const byte_t<2>>(data())[index];
        }
        return holder()->byte_2_value(index);
    }

    inline byte_t<4> byte_4_value(int64_t index) const
    {
        if (mode_ == mode_t::scalar_inplace) {
            ASSERT(index == 0);
            return base::span_cast<const byte_t<4>>(data())[index];
        }
        return holder()->byte_4_value(index);
    }

    inline byte_t<8> byte_8_value(int64_t index) const
    {
        if (mode_ == mode_t::scalar_inplace) {
            ASSERT(index == 0);
            return base::span_cast<const byte_t<8>>(data())[index];
        }
        return holder()->byte_8_value(index);
    }

    explicit array(std::string value)
        : dtype_(dtype::string)
        , mode_(mode_t::scalar_inplace)
        , has_data_(true)
    {
        const auto size = value.size();
        ASSERT(size <= s_array_bytes);
        data_.raw[0] = static_cast<char>(static_cast<uint8_t>(size));
        std::memcpy(&data_.raw[1], &value[0], size);
    }

public:
    array() = default;

    template <typename T>
    requires(base::arithmetic<T>)
    explicit array(T value)
        : dtype_(dtype_enum_v<T>)
        , mode_(mode_t::scalar_inplace)
        , has_data_(true)
    {
        ASSERT(dtype_is_numeric(dtype_));
        ASSERT(sizeof(T) < sizeof(data_));
        data_.raw[0] = static_cast<char>(static_cast<uint8_t>(sizeof(T)));
        std::memcpy(&data_.raw[1], &value, sizeof(T));
    }

    template <typename I>
    requires(!std::is_same_v<array, std::decay_t<I>> && !base::arithmetic<std::decay_t<I>>)
    explicit array(I&& impl)
        : dtype_(impl.dtype())
        , is_dynamic_(impl.is_dynamic())
        , has_data_(impl::has_data_member_function_v<std::remove_cvref_t<I>>)
        , has_eval_(impl::has_eval_member_function_v<std::remove_cvref_t<I>>)
        , has_copy_data_(impl::has_copy_data_member_function_v<std::remove_cvref_t<I>>)
        , has_stride_(impl::has_template_stride_member_function_v<std::remove_cvref_t<I>>)
    {
        if constexpr (impl::has_dimensions_member_function_v<std::remove_cvref_t<I>>) {
            dimensions_ = impl.dimensions();
        } else {
            ASSERT(!is_dynamic_);
            dimensions_ = static_cast<uint8_t>(impl.shape().size());
        }
        if constexpr (impl::has_is_none_member_variable_v<std::remove_cvref_t<I>>) {
            is_none_ = std::remove_cvref_t<I>::is_none;
        }
        using concrete_holder_type = concrete_holder_<std::remove_cvref_t<I>>;
        if constexpr (sizeof(concrete_holder_type) <= sizeof(data_)) {
            new (data_.get_raw()) concrete_holder_type(std::forward<I>(impl));
            mode_ = mode_t::holder_inplace;
        } else {
            auto p = std::make_shared<concrete_holder_type>(std::forward<I>(impl));
            new (data_.get_allocated()) holder_ptr(std::move(p));
            mode_ = mode_t::holder_pointer;
        }
    }

    array(const array& s)
        : dimensions_(s.dimensions_)
        , dtype_(s.dtype_)
        , mode_(s.mode_)
        , is_dynamic_(s.is_dynamic_)
        , has_data_(s.has_data_)
        , has_eval_(s.has_eval_)
        , has_copy_data_(s.has_copy_data_)
        , has_stride_(s.has_stride_)
        , is_none_(s.is_none_)
    {
        if (s.mode_ == mode_t::scalar_inplace) {
            std::memcpy(data_.raw, s.data_.raw, static_cast<uint8_t>(s.data_.raw[0]) + 1);
        } else if (s.mode_ == mode_t::holder_inplace) {
            s.holder()->copy_to(data_.get_raw());
        } else if (s.mode_ == mode_t::holder_pointer) {
            new (data_.get_allocated()) holder_ptr(*s.data_.get_allocated());
        }
    }

    array& operator=(const array& s)
    {
        if (this == &s) {
            return *this;
        }
        destroy();
        dimensions_ = s.dimensions_;
        mode_ = s.mode_;
        dtype_ = s.dtype_;
        is_dynamic_ = s.is_dynamic_;
        has_data_ = s.has_data_;
        has_eval_ = s.has_eval_;
        has_copy_data_ = s.has_copy_data_;
        has_stride_ = s.has_stride_;
        is_none_ = s.is_none_;
        if (s.mode_ == mode_t::scalar_inplace) {
            std::memcpy(data_.raw, s.data_.raw, static_cast<uint8_t>(s.data_.raw[0]) + 1);
        } else if (s.mode_ == mode_t::holder_inplace) {
            s.holder()->copy_to(data_.get_raw());
        } else if (s.mode_ == mode_t::holder_pointer) {
            new (data_.get_allocated()) holder_ptr(*s.data_.get_allocated());
        }
        return *this;
    }

    array(array&& s) noexcept
        : dimensions_(s.dimensions_)
        , dtype_(s.dtype_)
        , mode_(s.mode_)
        , is_dynamic_(s.is_dynamic_)
        , has_data_(s.has_data_)
        , has_eval_(s.has_eval_)
        , has_copy_data_(s.has_copy_data_)
        , has_stride_(s.has_stride_)
        , is_none_(s.is_none_)
    {
        if (s.mode_ == mode_t::scalar_inplace) {
            std::memcpy(data_.raw, s.data_.raw, static_cast<uint8_t>(s.data_.raw[0]) + 1);
        } else if (s.mode_ == mode_t::holder_inplace) {
            auto p = data_.raw;
            s.holder()->move_to(p);
        } else if (s.mode_ == mode_t::holder_pointer) {
            new (data_.get_allocated()) holder_ptr(std::move(*s.data_.get_allocated()));
        }
    }

    array& operator=(array&& s) noexcept
    {
        if (this == &s) {
            return *this;
        }
        destroy();
        dimensions_ = s.dimensions_;
        mode_ = s.mode_;
        dtype_ = s.dtype_;
        is_dynamic_ = s.is_dynamic_;
        has_data_ = s.has_data_;
        has_eval_ = s.has_eval_;
        has_copy_data_ = s.has_copy_data_;
        has_stride_ = s.has_stride_;
        is_none_ = s.is_none_;
        if (s.mode_ == mode_t::scalar_inplace) {
            std::memcpy(data_.raw, s.data_.raw, static_cast<uint8_t>(s.data_.raw[0]) + 1);
        } else if (s.mode_ == mode_t::holder_inplace) {
            s.holder()->move_to(data_.get_raw());
        } else if (s.mode_ == mode_t::holder_pointer) {
            new (data_.get_allocated()) holder_ptr(std::move(*s.data_.get_allocated()));
        }
        return *this;
    }

    ~array()
    {
        destroy();
    }

    explicit operator bool() const noexcept
    {
        return mode_ != mode_t::null;
    }

    inline enum dtype dtype() const noexcept
    {
        ASSERT(mode_ != mode_t::null);
        return dtype_;
    }

    inline uint32_t dimensions() const noexcept
    {
        ASSERT(mode_ != mode_t::null);
        return dimensions_;
    }

    inline uint32_t size() const
    {
        auto s = shape();
        if (s.empty()) {
            throw invalid_operation("Can't get size of scalar array.");
        }
        return static_cast<uint32_t>(s[0]);
    }

    inline bool empty() const
    {
        if (is_dynamic()) {
            return (size() == 0);
        }
        if (has_data_) {
            return data().empty();
        }
        auto sh = shape();
        return std::ranges::find(sh, 0u) != sh.end();
    }

    inline uint64_t volume() const
    {
        if (is_dynamic()) {
            auto s = shape()[0];
            uint64_t v = 0;
            for (int64_t i = 0L; i < s; ++i) {
                v += operator[](i).volume();
            }
            return v;
        }
        if (has_data_) {
            return data().size() / nd::dtype_bytes(dtype_);
        }
        auto sh = shape();
        return std::accumulate(sh.begin(), sh.end(), 1L, std::multiplies<uint64_t>());
    }

    template <typename T>
    inline T value(int64_t index) const
    {
        if constexpr (!dtype_is_numeric(dtype_enum_v<T>)) {
            throw non_numeric_dtype(dtype_to_str(dtype_enum_v<T>));
        } else {
            if (is_dynamic()) {
                return dynamic_value<T>(index);
            }
            ASSERT(index >= 0);
            ASSERT(index < volume());
            switch (dtype()) {
            case dtype::boolean:
                return static_cast<T>(base::bit_cast<bool>(byte_1_value(index)));
            case dtype::uint8:
                return static_cast<T>(base::bit_cast<uint8_t>(byte_1_value(index)));
            case dtype::int8:
                return static_cast<T>(base::bit_cast<int8_t>(byte_1_value(index)));
            case dtype::uint16:
                return static_cast<T>(base::bit_cast<uint16_t>(byte_2_value(index)));
            case dtype::int16:
                return static_cast<T>(base::bit_cast<int16_t>(byte_2_value(index)));
            case dtype::uint32:
                return static_cast<T>(base::bit_cast<uint32_t>(byte_4_value(index)));
            case dtype::int32:
                return static_cast<T>(base::bit_cast<int32_t>(byte_4_value(index)));
            case dtype::uint64:
                return static_cast<T>(base::bit_cast<uint64_t>(byte_8_value(index)));
            case dtype::int64:
                return static_cast<T>(base::bit_cast<int64_t>(byte_8_value(index)));
            case dtype::bfloat16:
                return static_cast<T>(base::bit_cast<base::bfloat16>(byte_2_value(index)));
            case dtype::float16:
                return static_cast<T>(base::bit_cast<base::half>(byte_2_value(index)));
            case dtype::float32:
                return static_cast<T>(base::bit_cast<float>(byte_4_value(index)));
            case dtype::float64:
                return static_cast<T>(base::bit_cast<double>(byte_8_value(index)));
            case dtype::string:
                throw non_numeric_dtype(dtype_to_str(dtype::string));
            case dtype::object:
                throw non_numeric_dtype(dtype_to_str(dtype::object));
            case dtype::byte:
                throw non_numeric_dtype(dtype_to_str(dtype::byte));
            case dtype::unknown:
                throw unknown_dtype();
            }
            return T();
        }
    }

    dict dict_value(int64_t index) const;

    inline icm::shape shape() const
    {
        if (mode_ == mode_t::scalar_inplace) {
            return icm::shape();
        }
        check_null();
        return holder()->shape();
    }

    inline bool has_data() const noexcept
    {
        ASSERT(mode_ != mode_t::null);
        return has_data_;
    }

    inline std::span<const uint8_t> data() const
    {
        if (mode_ == mode_t::scalar_inplace) {
            return base::span_cast<const uint8_t>(
                std::span<const char>(&data_.raw[1], static_cast<uint8_t>(data_.raw[0])));
        }
        check_null();
        return holder()->data();
    }

    inline bool has_stride() const noexcept
    {
        ASSERT(mode_ != mode_t::null);
        return has_stride_;
    }

    template <typename I>
    array stride(icm::index_mapping_t<I>&& source_shape) const;

    template <typename T>
    inline std::span<const T> data() const
    {
        ASSERT(has_data());
        auto sp = data();
        return base::span_cast<const T>(sp);
    }

    template <typename F>
    inline auto with_data(F f) const
    {
        ASSERT(has_data());
        return switch_dtype(dtype(), [this, f]<typename T>() {
            auto sp = data();
            auto d = base::span_cast<const T>(sp);
            return f.template operator()<T>(d);
        });
    }

    template <typename F>
    inline auto with_numeric_data(F f) const
    {
        ASSERT(has_data());
        return switch_numeric_dtype(dtype(), [this, f]<typename T>() {
            auto sp = data();
            auto d = base::span_cast<const T>(sp);
            return f.template operator()<T>(d);
        });
    }

    inline iterator begin() const
    {
        check_null();
        auto h = holder();
        if (h->shape().empty()) {
            throw invalid_operation("Can't iterate over scalar array.");
        }
        try {
            return h->begin();
        } catch (const invalid_operation&) {
            return icm::index_based_iterator<array, array, icm::use_container_index_tag, int64_t>(*this, 0L);
        }
        ASSERT(false);
        return iterator{static_cast<array*>(nullptr)};
    }

    inline iterator end() const
    {
        check_null();
        auto h = holder();
        if (h->shape().empty()) {
            throw invalid_operation("Can't iterate over scalar array.");
        }
        try {
            return h->end();
        } catch (const invalid_operation&) {
            return icm::index_based_iterator<array, array, icm::use_container_index_tag, int64_t>(*this, size());
        }
        ASSERT(false);
        return iterator{static_cast<array*>(nullptr)};
    }

    inline array operator[](int64_t index) const
    {
        check_null();
        auto h = holder();
        if (h->shape().empty()) {
            throw invalid_operation("Can't subscript scalar array.");
        }
        return h->get(index);
    }

    inline array range(int64_t start, int64_t end) const
    {
        try {
            return holder()->get_range(start, end);
        } catch (const invalid_operation&) {
            return array(impl::range_array(*this, start, end));
        }
        ASSERT(false);
        return array{};
    }

    inline constexpr bool is_dynamic() const noexcept
    {
        return is_dynamic_;
    }

    inline bool is_none() const noexcept
    {
        return is_none_;
    }

private:
    template <typename T>
    inline T dynamic_value(int64_t index) const
    {
        ASSERT(index < volume());
        auto v = 0;
        auto i = 0;
        while (v <= index) {
            auto a = operator[](i);
            if (v + a.volume() > index) {
                return a.value<T>(index - v);
            } else {
                v += a.volume();
                ++i;
            }
        }
        ASSERT(false);
        return T();
    }

    template <typename T>
    inline const T* dynamic_cast_() const
    {
        const auto* h = dynamic_cast<const concrete_holder_<T>*>(holder());
        if (h == nullptr) {
            return nullptr;
        }
        return &h->impl_;
    }

    template <typename T>
    inline const T* static_cast_() const
    {
        const auto* h = static_cast<const concrete_holder_<T>*>(holder());
        return &h->impl_;
    }

    inline holder_* holder() noexcept
    {
        switch (mode_) {
        case mode_t::null:
        case mode_t::scalar_inplace:
            return nullptr;
        case mode_t::holder_inplace:
            return reinterpret_cast<holder_*>(data_.get_raw());
        case mode_t::holder_pointer:
            return data_.get_allocated()->get();
        }
        return nullptr;
    }

    inline const holder_* holder() const
    {
        return const_cast<array*>(this)->holder();
    }

    inline void check_null() const
    {
        if (mode_ == mode_t::null) [[unlikely]] {
            throw invalid_operation("Null array");
        }
    }

    inline void destroy()
    {
        switch (mode_) {
        case mode_t::null:
        case mode_t::scalar_inplace:
            return;
        case mode_t::holder_inplace:
            reinterpret_cast<holder_*>(data_.get_raw())->~holder_();
            return;
        case mode_t::holder_pointer:
            data_.get_allocated()->~holder_ptr();
            return;
        }
    }

    inline bool has_eval() const noexcept
    {
        return has_eval_;
    }

    inline bool has_copy_data() const noexcept
    {
        return has_copy_data_;
    }

    inline array eval() const
    {
        ASSERT(has_eval_);
        return holder()->eval();
    }

    inline void copy_data(std::span<uint8_t> buffer) const
    {
        ASSERT(has_copy_data_);
        return holder()->copy_data(buffer);
    }

    inline void nonzero(icm::bit_vector_view output) const
    {
        if (has_data_) {
            nd::switch_numeric_dtype(dtype(), [this, &output]<typename T>() {
                auto y = data<T>();
                output.set_from_span(y);
            });
            return;
        }
        try {
            holder()->nonzero(output);
        } catch (const invalid_operation&) {
            int64_t i = 0L;
            auto e = end();
            for (auto it = begin(); it != e; ++it) {
                try {
                    auto d = *it;
                    if (d && !d.empty() && d.value<bool>(0)) {
                        output.set(i);
                    }
                } catch (const std::exception&) {
                    /// Intentionally empty.
                }
                ++i;
            }
        }
    }

    friend array eval(array);
    friend void copy_data(const array& arr, std::span<uint8_t>);
    friend void nonzero(const array& arr, icm::bit_vector_view output);
    friend array adapt(std::string str);

    template <typename T>
    friend const T* dynamic_array_cast(const array& arr);

    template <typename T>
    friend const T* static_array_cast(const array& arr);

private:
    union data
    {
        data()
        {
        }
        ~data()
        {
        }

        void_ptr get_raw() const noexcept
        {
            return std::launder(const_cast<data*>(this)->raw);
        }

        holder_ptr* get_allocated() const noexcept
        {
            return std::launder(&const_cast<data*>(this)->allocated);
        }

        char raw[s_array_bytes + 1];
        holder_ptr allocated;
    } data_;
    uint8_t dimensions_ = 0;
    enum dtype dtype_ = nd::dtype::unknown;
    enum mode_t mode_ = mode_t::null;
    bool is_dynamic_ = false;
    bool has_data_ = false;
    bool has_eval_ = false;
    bool has_copy_data_ = false;
    bool has_stride_ = false;
    bool is_none_ = false;
};

} // namespace nd

#include "dict.hpp"

namespace nd {

template <typename I>
dict array::concrete_holder_<I>::dict_value(int64_t index) const
{
    if constexpr (impl::has_value_member_function_v<I, dict>) {
        ASSERT(impl_.dtype() == dtype::object);
        return impl_.value(index);
    } else {
        throw invalid_operation("Can't convert array value to dict.");
    }
}

template <typename I>
array array::concrete_holder_<I>::get(int64_t index) const
{
    if constexpr (impl::has_get_member_function_v<I>) {
        return impl_.get(index);
    } else if constexpr (!impl::is_scalar_v<I> && !impl::has_is_none_member_variable_v<I>) {
        auto s = impl_.shape();
        ASSERT(!s.empty());
        auto v =
            std::accumulate(s.begin() + 1, s.end(), 1UL, std::multiplies<uint64_t>()) * nd::dtype_bytes(impl_.dtype());
        auto d = impl_.data().data();
        static_assert(impl::has_owner_member_function_v<I>);
        return array(impl::std_span_array_nd(impl_.owner(),
                                             std::span<const uint8_t>(d + index * v, v),
                                             icm::shape(s.begin() + 1, s.end()),
                                             impl_.dtype()));
    }
    ASSERT_MESSAGE(false, "Can't get value from scalar array.");
    return array();
}

template <typename I>
array array::concrete_holder_<I>::get_range(int64_t start, int64_t end) const
{
    if constexpr (impl::has_data_member_function_v<I> && !impl::is_scalar_v<I> &&
                  !impl::has_is_none_member_variable_v<I>) {
        auto s = impl_.shape();
        ASSERT(!s.empty());
        auto v =
            std::accumulate(s.begin() + 1, s.end(), 1UL, std::multiplies<uint64_t>()) * nd::dtype_bytes(impl_.dtype());
        auto d = impl_.data().data();
        auto sh = icm::small_vector<int64_t>(s.begin(), s.end());
        sh[0] = end - start;
        static_assert(impl::has_owner_member_function_v<I>);
        return array(impl::std_span_array_nd(
            impl_.owner(), std::span<const uint8_t>(d + start * v, (end - start) * v), icm::shape(sh), impl_.dtype()));
    } else {
        throw invalid_operation("range() method is not implemented for this array.");
    }
}

} // namespace nd

#include "impl/iterator_impl.hpp"
