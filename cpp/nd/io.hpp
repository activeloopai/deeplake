#pragma once

/**
 * @file io.hpp
 * @brief Definitions of input/output operators and `save`/`load` functions.
 */

#include "array.hpp"

#include <base/memory_buffer.hpp>
#include <filesystem>
#include <format/sequence_index.hpp>
#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace nd {

struct array_header
{
    std::vector<std::pair<std::string, nd::dtype>> keys_;
    std::vector<icm::shape> shapes_;
    std::vector<uint64_t> offsets_;
    std::vector<uint8_t> is_dynamic_;
    int64_t data_offset_ = 0;
    int64_t num_rows_ = 0;
    nd::dtype dtype_ = nd::dtype::unknown;
    bool is_dynamic_array_ = false;
    bool has_schema_ = false;
    uint32_t version_ = 1U;
    std::vector<uint64_t> dynamic_rows_ = {1};

    void read_top_data(const base::memory_buffer& buffer);
    void read_header(const base::memory_buffer& buffer);
    nd::array get(const base::memory_buffer& buffer, nd::dtype dt, int64_t index) const;
};

struct header_info
{
    icm::shape shape_;
    int64_t header_size_ = 0;
    bool is_static_ = false;
    bool has_schema_ = false;

    std::optional<format::sequence_index> sequence_index_ = std::nullopt;

    bool operator==(const header_info& other) const
    {
        return shape_ == other.shape_ && header_size_ == other.header_size_ && is_static_ == other.is_static_ &&
               has_schema_ == other.has_schema_;
    }
};

/**
 * @fn void write_to(array, std::vector<uint8_t>&)
 * @brief Writes the array into the buffer.
 *
 * @param arr Array to write.
 * @param buffer Buffer to write into.
 * @param has_schema Specifies whether the object type of array has schema.
 * @return The shape of array if array is static, the number of bytes of header and whether the array is static.
 */
header_info write_to(const array& arr, std::vector<uint8_t>& buffer, bool has_schema = false);

/**
 * @fn nd::array read_from(const base::memory_buffer&)
 * @brief Reads an array from the buffer.
 *
 * @param buffer Buffer to read from.
 * @return The read array.
 */
nd::array read_from(const base::memory_buffer& buffer);

/**
 * @brief Writes an array into file.
 *
 * @param a Array to write.
 * @param path File path.
 */
void save(const nd::array& a, const std::filesystem::path& p);

/**
 * @brief Appends the given subarrays to the array cached at the specified path.
 *
 * @param path Cached array path.
 * @param arrays Subarrays to append.
 */
void append_arrays(const std::filesystem::path& p, const nd::array& arrays);

/**
 * @brief Removes the given subarrays from the array cached at the specified path.
 *
 * @param path Cached array path.
 * @param indices Indices of the subarrays to remove.
 * @pre Indices should be sorted ascending.
 */
void remove_arrays(const std::filesystem::path& p, std::vector<int64_t> indices);

/**
 * @brief Updates the given subarray in the array cached at the specified path.
 *
 * @param path Cached array path.
 * @param arrays Subarrays to update.
 * @param indices Indices of the subarrays to update.
 * @pre Indices should be sorted ascending.
 */
void update_arrays(const std::filesystem::path& p, const nd::array& arrays, std::vector<int64_t> indices);

/**
 * @brief Loads an array from the file.
 *
 * @param path File path
 * @param mmap If true, memory mapping is used.
 * @param is_dynamic If true, the values have dynamic length.
 * @return array The loaded array.
 */
array load(const std::filesystem::path& path, bool mmap, bool is_dynamic = false);

/**
 * @brief Loads an array from the stream.
 *
 * @param stream stream
 * @return array The loaded array.
 */
array load(std::istream& stream);

/**
 * @brief Loads an array from the numpy file.
 *
 * @param stream Stream to the numpy file or buffer.
 * @return array The loaded array.
 */
array load_from_npy(std::istream& stream);

std::ostream& operator<<(std::ostream& os, const array& a);

std::string to_sql_string(const array&);

icm::json to_json(const array& a);

array from_json(const icm::const_json& json);

} // namespace nd