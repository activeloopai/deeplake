#pragma once

#include <format/serializable.hpp>
#include <format/pod_serializable.hpp>
#include <format/serializer.hpp>
#include <icm/shape.hpp>

namespace format {

template <>
struct serializable<icm::shape>
{
    inline static int64_t output_size(const icm::shape& shape) noexcept
    {
        return sizeof(shape.size()) + (shape.size() * sizeof(icm::shape::value_type));
    }

    static read_result<icm::shape> read(const base::memory_buffer& buffer, int64_t offset)
    {
        int64_t size = 0;
        std::tie(size, offset) = format::read<int64_t>(buffer, offset);
        icm::small_vector<icm::shape::value_type> shape;
        shape.reserve(size);
        for (auto i = 0; i < size; ++i) {
            shape.emplace_back(0);
            std::tie(shape.back(), offset) = format::read<icm::shape::value_type>(buffer, offset);
        }
        return {icm::shape(shape), offset};
    }

    static void write(const icm::shape& shape, buffer_t& bytes, int64_t offset)
    {
        offset = format::write(shape.size(), bytes, offset);
        for (const auto& dim : shape) {
            offset = format::write(dim, bytes, offset);
        }
    }
};

}
