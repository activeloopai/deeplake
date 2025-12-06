#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace deeplake {

inline const std::string UNCOMMITTED_VERSION = "uncommitted";

enum class indexing_mode
{
    always,     // Indexing is aligned with commit blocks
    automatic,  // Indexing is not aligned with commit blocks, but with recommended size
    off         // Indexing is disabled
};

using datafile_field_location_t = std::string;
using block_id_t = uint32_t;
using block_offset_t = uint32_t;
using row_id_t = uint64_t;

inline row_id_t to_row_id(block_id_t block_id, block_offset_t offset) noexcept
{
    return (static_cast<row_id_t>(block_id) << 32) | offset;
}

inline block_id_t get_block_id(row_id_t row_id) noexcept
{
    return static_cast<block_id_t>(row_id >> 32);
}

inline block_offset_t get_offset(row_id_t row_id) noexcept
{
    return static_cast<block_offset_t>(row_id);
}

inline std::pair<block_id_t, block_offset_t> split_row_id(row_id_t row_id) noexcept
{
    return {get_block_id(row_id), get_offset(row_id)};
}

} // namespace deeplake
