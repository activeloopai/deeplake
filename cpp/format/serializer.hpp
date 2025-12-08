#pragma once

#include "buffer.hpp"
#include "impl/mpl.hpp"
#include "read_result.hpp"
#include "serializable.hpp"

#include <base/memory_buffer.hpp>
#include <icm/trie.hpp>

#include <cstdint>

namespace format {

template <typename T>
struct serializer
{
    static_assert(sizeof(serializable<T>) > 0, "the specialization serializable<T> must be included in places where serializer.hpp is used");
    /**
     * @brief Constructs empty object for read.
     * 
     * @return T The constructed object.
     */
    inline static T construct()
    {
        if constexpr (impl::has_construct_member_function_v<serializable<T>>) {
            return serializable<T>::construct();
        } else {
            return T();
        }
    }
    /**
     * @brief Reads the object from the buffer.
     * 
     * @param bytes Buffer to read from.
     * @return read_result<T> Result of the read.
     */
    inline static read_result<T> read(const base::memory_buffer& bytes, int64_t offset)
    {
        return std::move(serializable<T>::read(bytes, offset));
    }

    /**
     * @brief Writes the given object in the given buffer at the given offset.
     * Returns the new offset end of the written data.
     * 
     * @param o Object.
     * @param bytes Buffer to write.
     * @param offset Offset to write at.
     * @return int64_t The new offset.
     */
    inline static int64_t write(const T& o, buffer_t& bytes, int64_t offset)
    {
        if constexpr (impl::has_output_size_member_function_v<serializable<T>>) {
            auto size = serializable<T>::output_size(o);
            bytes.resize(std::max(offset + size, static_cast<int64_t>(bytes.size())));
            serializable<T>::write(o, bytes, offset);
            return offset + size;
        } else {
            auto buf = serializable<T>::write(o);
            bytes.insert(bytes.begin() + offset, std::make_move_iterator(buf.begin()),
                         std::make_move_iterator(buf.end()));
            return offset + buf.size();
        }
    }
};

template <typename T>
inline read_result<T> read(const base::memory_buffer& bytes, int64_t offset)
{
    return serializer<T>::read(bytes, offset);
}

template <typename T>
int64_t write(const T& o, buffer_t& bytes, int64_t offset)
{
    return serializer<T>::write(o, bytes, offset);
}

}
