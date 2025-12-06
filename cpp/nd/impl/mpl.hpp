#pragma once

#include <icm/const_json.hpp>
#include <icm/index_mapping.hpp>

#include <base/function.hpp>
#include <cstdint>
#include <functional>
#include <span>

namespace icm {
class bit_vector_view;
} // namespace icm

namespace nd {

class array;
class dict;

namespace impl {

template <typename T>
struct has_data_member_function
{
    template <typename U>
    static int check(decltype(static_cast<std::span<const uint8_t> (U::*)() const>(&U::data))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_data_member_function_v = has_data_member_function<T>::value;

template <typename T>
struct has_dict_data_member_function
{
    template <typename U>
    static int check(decltype(static_cast<icm::const_json (U::*)() const>(&U::data))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_dict_data_member_function_v = has_dict_data_member_function<T>::value;

template <typename T>
struct has_serialize_member_function
{
    template <typename U>
    static int check(decltype(static_cast<std::string (U::*)() const>(&U::serialize))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_serialize_member_function_v = has_serialize_member_function<T>::value;

template <typename T>
struct has_get_member_function
{
    template <typename U>
    static int check(decltype(static_cast<nd::array (U::*)(int64_t) const>(&U::get))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_get_member_function_v = has_get_member_function<T>::value;

template <typename T, typename V>
struct has_value_member_function
{
    template <typename U>
    static int check(decltype(static_cast<V (U::*)(int64_t) const>(&U::value))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T, typename V>
constexpr bool has_value_member_function_v = has_value_member_function<T, V>::value;

template <typename T, typename V>
struct has_dict_value_member_function
{
    template <typename U>
    static int check(decltype(static_cast<V (U::*)(int64_t, dict*) const>(&U::dict_value))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T, typename V>
constexpr bool has_dict_value_member_function_v = has_dict_value_member_function<T, V>::value;

template <typename T>
struct has_eval_member_function
{
    template <typename U>
    static int check(decltype(static_cast<nd::array (U::*)() const>(&U::eval))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_eval_member_function_v = has_eval_member_function<T>::value;

template <typename T>
struct has_dict_eval_member_function
{
    template <typename U>
    static int check(decltype(static_cast<nd::dict (U::*)() const>(&U::eval))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_dict_eval_member_function_v = has_dict_eval_member_function<T>::value;

template <typename T>
struct has_copy_data_member_function
{
    template <typename U>
    static int check(decltype(static_cast<void (U::*)(std::span<uint8_t>) const>(&U::copy_data))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_copy_data_member_function_v = has_copy_data_member_function<T>::value;

template <typename T>
struct has_nonzero_member_function
{
    template <typename U>
    static int check(decltype(static_cast<void(U::*)(icm::bit_vector_view) const>(&U::nonzero))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_nonzero_member_function_v = has_nonzero_member_function<T>::value;

template <typename T>
struct has_is_none_member_variable
{
    template <typename U>
    static auto check(int) -> decltype(U::is_none, std::true_type{});

    template <typename U>
    static std::false_type check(...);

     static constexpr bool value = decltype(check<T>(0))::value;
};

template <typename T>
constexpr bool has_is_none_member_variable_v = has_is_none_member_variable<T>::value;

template <typename T>
struct has_dimensions_member_function
{
    template <typename U>
    static int check(decltype(static_cast<uint8_t (U::*)() const>(&U::dimensions))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_dimensions_member_function_v = has_dimensions_member_function<T>::value;

template <typename T>
struct has_begin_member_function
{
    template <typename U>
    static int check(decltype(&U::begin)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_begin_member_function_v = has_begin_member_function<T>::value;

template <typename T>
struct has_end_member_function
{
    template <typename U>
    static int check(decltype(&U::end)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_end_member_function_v = has_end_member_function<T>::value;

template <typename T>
struct has_foreach_subarray_member_function
{
    template <typename U>
    static int
    check(decltype(static_cast<void (U::*)(const std::function<void(const nd::array&)>&) const>(&U::foreach))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_foreach_subarray_member_function_v = has_foreach_subarray_member_function<T>::value;

template <typename T>
struct has_foreach_subarray_member_function_with_exception
{
    template <typename U>
    static int check(
        decltype(static_cast<void (U::*)(const std::function<void(const std::variant<nd::array, std::exception_ptr>&)>&)
                                 const>(&U::foreach))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_foreach_subarray_member_function_with_exception_v =
    has_foreach_subarray_member_function_with_exception<T>::value;

template <typename T>
struct has_all_of_subarray_member_function
{
    template <typename U>
    static int
    check(decltype(static_cast<bool (U::*)(const std::function<bool(const nd::array&)>&) const>(&U::all_of))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_all_of_subarray_member_function_v = has_all_of_subarray_member_function<T>::value;

template <typename T>
struct has_any_of_subarray_member_function
{
    template <typename U>
    static int
    check(decltype(static_cast<bool (U::*)(const std::function<bool(const nd::array&)>&) const>(&U::any_of))*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_any_of_subarray_member_function_v = has_any_of_subarray_member_function<T>::value;

template <typename T>
struct has_owner_member_function
{
    template <typename U>
    static int check(decltype(&U::owner)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_owner_member_function_v = has_owner_member_function<T>::value;

template <typename T>
struct has_subscript_member_operator
{
    template <typename U>
    static auto check(decltype(std::declval<U&>()[std::declval<std::string>()])*) -> std::true_type;

    template <typename U>
    static std::false_type check(...);

    static constexpr bool value = decltype(check<T>(0))::value;
};

template <typename T>
constexpr bool has_subscript_member_operator_v = has_subscript_member_operator<T>::value;

template <typename T>
struct has_keys_member_function
{
    template <typename U>
    static int check(decltype(&U::keys)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_keys_member_function_v = has_keys_member_function<T>::value;

template <typename T>
struct has_template_stride_member_function
{
    template <typename U, template <typename...> class>
    static int check(U* u, decltype(&U::template stride<int>)* = nullptr);

    template <typename, template <typename...> class>
    static char check(...);

    static constexpr bool value = sizeof(check<T, std::void_t>(nullptr)) == sizeof(int);
};

template <typename T>
constexpr bool has_template_stride_member_function_v = has_template_stride_member_function<T>::value;

template <typename T>
struct has_contains_member_function
{
    template <typename U>
    static int check(decltype(&U::contains)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_contains_member_function_v = has_contains_member_function<T>::value;

template <typename T>
struct has_is_scalar_tag
{
    template <typename U>
    static int check(decltype(&U::is_scalar_tag)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(0)) == sizeof(int));
};

template <typename T>
constexpr bool is_scalar_v = has_is_scalar_tag<T>::value;

} // namespace impl

} // namespace nd
