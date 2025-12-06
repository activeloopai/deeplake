#pragma once

#include "exceptions.hpp"
#include "index_mapping.hpp"
#include "slice.hpp"
#include "small_vector.hpp"

#include <string>
#include <variant>

namespace icm {

template <typename T>
using indexable_t = std::variant<slice_t<T>, index_mapping_t<T>>;

template <typename T>
using indexable_vector_t = small_vector<indexable_t<T>>;

using indexable = indexable_t<int64_t>;
using indexable_vector = indexable_vector_t<int64_t>;

/**
 * @brief Computes the `index_mapping` from the given `indexable` with the given desired size.
 * 
 * @tparam enforce_size If true, throws an exception when index contains value outside of given size
 * @tparam T Index type
 * @param index indexable object
 * @param size Desired size
 * @return index_mapping_t<T> 
 */
template <typename T, bool enforce_size = false>
index_mapping_t<T> compute_index_mapping(const indexable_t<T>& index, T size)
{
    auto enforce = [](const index_mapping_t<T>& mapping, uint32_t e_size) {
        for (auto i = 0; i < mapping.size(); ++i) {
            if (mapping[i] >= e_size) {
                throw exception("Index \"" + std::to_string(mapping[i]) + "\" is out of range 0-" +
                                std::to_string(e_size));
            }
        }
    };

    return std::visit(
        [size, enforce] <typename V>(V& arg) {
            if constexpr (std::is_same_v<V, const slice_t<T>>) {
                return arg.compute(size);
            } else if constexpr (std::is_same_v<V, const index_mapping_t<T>>) {
                auto ii = arg;
                if constexpr (enforce_size) {
                    enforce(ii, size);
                }
                return ii;
            } else {
                ASSERT_MESSAGE(false, "Invalid index");
                return index_mapping_t<T>{};
            }
        },
        index);
}

}
