#pragma once

namespace base {

/**
 * @brief A simple overload helper for lambdas.
 * This class is mainly used to create a lambda that can handle multiple types. The main usage is std::visit for
 * variants.
 */
template <typename... Ts>
struct overloads : Ts...
{
    using Ts::operator()...;
};

template <typename... Ts>
overloads(Ts...) -> overloads<Ts...>; // deduction guide for pre c++20 compilers

}
