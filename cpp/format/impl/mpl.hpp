#pragma once

namespace format::impl {

template <typename T>
struct has_output_size_member_function
{

    template <typename U>
    static int check(decltype(&U::output_size)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_output_size_member_function_v = has_output_size_member_function<T>::value;

template <typename T>
struct has_construct_member_function
{

    template <typename U>
    static int check(decltype(&U::construct)*);

    template <typename U>
    static char check(...);

    static bool constexpr value = (sizeof(check<T>(nullptr)) == sizeof(int));
};

template <typename T>
constexpr bool has_construct_member_function_v = has_construct_member_function<T>::value;

}