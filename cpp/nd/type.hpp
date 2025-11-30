#pragma once

/**
 * @file type.hpp
 * @brief Declaration of the type interface.
 */

#include "dtype.hpp"
#include "scalar_type.hpp"

#include <icm/shape.hpp>

#include <string>

namespace nd {

/**
 * @brief Base interface for all types in the deeplake.
 */
class type
{
public:
    /**
     * @brief Construct a scalar type from dtype.
     * @param scalar_type The data type of the scalar.
     */
    static type scalar(scalar_type dtype)
    {
        return type(dtype);
    }

    /**
     * @brief Construct an array type from dtype and shape.
     * dimension is computed from shape.
     * @param scalar_type The data type of the array.
     * @param shape The shape of the array.
     */
    static type array(scalar_type dtype, icm::shape shape)
    {
        return type(dtype, std::move(shape));
    }

    /**
     * @brief Construct an array type from dimensions.
     * @param scalar_type The data type of the array.
     * @param dimensions The number of dimensions of the array.
     */
    static type array(scalar_type dtype, uint8_t dimensions)
    {
        return type(dtype, dimensions);
    }

    /**
     * @brief Construct an array type from dtype with unknown dimensions.
     * @param scalar_type The data type of the array.
     */
    static type array(scalar_type dtype)
    {
        auto r = type(dtype);
        r.shape_ = icm::shape::unknown();
        return r;
    }

    /**
     * @brief Construct an array type from type adding a dimension.
     * @param type The reference type.
     * @param first_dimension The size of the first dimension.
     * @return The new array type.
     */
    static type array(const type& tp, int64_t first_dimension)
    {
        const auto& sh = tp.get_shape();
        std::vector<int64_t> shape{first_dimension};
        shape.insert(shape.end(), sh.begin(), sh.end());
        return type::array(tp.get_scalar_type(), icm::shape(std::move(shape)));
    }

    /**
     * @brief Construct a dtype casted type from type,
     * which can be used to override dtype of a type.
     * @param t The reference type.
     * @param dtype The dtype to cast to.
     */
    static type cast(const type& t, scalar_type dtype)
    {
        return type(t, dtype);
    }

    static type unknown()
    {
        return type(scalar_type(dtype::unknown));
    }

    /// Construct a type from a string representation of dtype.
    static type from_dtype_string(std::string_view dtype_str);

    /**
     * @brief Construct a type from JSON.
     * @param j The JSON object.
     */
    static type from_json(const icm::const_json& j);

    /**
     * @brief Get the name of the type.
     * @return The name of the type.
     */
    std::string id() const noexcept;

    /**
     * @brief Serialize the type to JSON.
     */
    [[nodiscard]] icm::json to_json() const;

    /**
     * @brief Serialize the type to a string.
     */
    [[nodiscard]] std::string to_string(bool indent = false) const;

   /**
     * @brief Get the data type of the type.
     * @return The data type of the type.
     */
    inline enum dtype get_dtype() const noexcept
    {
        return get_scalar_type().get_dtype();
    }

    inline const scalar_type& get_scalar_type() const noexcept
    {
        return scalar_type_;
    }

    inline bool is_numeric() const noexcept
    {
        return dtype_is_numeric(scalar_type_.get_dtype());
    }

    inline bool is_static() const noexcept
    {
        return shape_.is_known();
    }

    inline bool is_array() const noexcept
    {
        return !shape_.empty();
    }

    /**
     * @brief Get the shape of the type, if it has one. Otherwise, return nullopt.
     * @return The shape of the type.
     */
    const icm::shape& get_shape() const noexcept
    {
        return shape_;
    }

    /**
     * @brief Get the number of dimensions of the type.
     * @return The number of dimensions of the type.
     */
    uint8_t dimensions() const noexcept
    {
        if (shape_.is_unknown()) {
            return unknown_dimensions;
        }
        return static_cast<uint8_t>(shape_.size());
    }

    /**
     * @brief validate if the provided array meets the type requirements(dimensions).
     *
     * @param arr input array
     */
    void validate(const class array& arr) const;

    /**
     * @brief validate if the provided batch meets the type requirements(dimensions).
     *
     * @param arr input batch
     */
    void validate_batch(const class array& batch) const;

    /**
     * @brief casts the provided array to the type requirements.
     * @param arr input array
     * @return The casted array.
     */
    class array cast_array(const class array& arr) const;

    /**
     * @brief casts the provided batch to the type requirements.
     * @param arr input batch
     * @return The casted batch.
     */
    class array cast_batch(const class array& batch) const;

    bool operator==(const type& other) const
    {
        return scalar_type_ == other.scalar_type_ && shape_ == other.shape_;
    }

    /**
     * @brief Get the array representing none value of the type.
     * @return The array representing none value of the type.
     */
    class array none_value() const;

    /**
     * @brief Get the array representing default value of the type.
     * @return The array representing default value of the type.
     */
    class array default_value() const;

    constexpr static uint8_t unknown_dimensions = 255;

    type()
        : type(scalar_type(dtype::unknown))
    {
    }

private:
    void validate_schema(const class array& arr) const;

    class array cast_schema(const class array& arr) const;

    explicit type(scalar_type dtype);

    type(scalar_type dtype, icm::shape shape);

    type(scalar_type dtype, uint8_t dimensions);

    type(const type& t, scalar_type dtype);

    explicit type(const icm::const_json& j);

private:
    /// If shape is not set, then the dimensions are unknown.
    icm::shape shape_;
    scalar_type scalar_type_;
};

type slice_type(const type& source, uint8_t slices);
type slice_type(const type& source, const icm::shape& shape);

} // namespace nd
