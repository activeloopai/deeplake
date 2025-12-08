#pragma once

/**
 * @file reader.hpp
 * @brief Definition of `reader` class.
 */

#include "exceptions.hpp"
#include "fetch_options.hpp"
#include "provider_base.hpp"
#include "resource_meta.hpp"
#include "storage.hpp"

#include <async/promise.hpp>
#include <base/read_buffer.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <exception>
#include <functional>
#include <future>
#include <string>

namespace storage {

class writer;


class reader : public virtual provider_base
{
public:
    /**
     * @brief Get the length of the resource, if it exists.
     *
     * @param path The path of the resource to check.
     * @return promise<int> The length of the resource. Returns -1 if it does not exist.
     */
    virtual async::promise<int64_t> length(const std::string& path) const = 0;

    /**
     * @brief list the content of the provided bucket. Returned resources are sorted lexicographically. If the location
     * does not exist, return an empty vector.
     *
     * @param prefix Returns objects from the given prefix only.
     */
    virtual async::promise<std::vector<resource_meta>> list(const std::string& prefix) const
    {
        throw reader_error{prefix, 501, "not_implemented"};
    }

    /**
     * @brief list only the immediate subdirectories (common prefixes) at the given prefix level.
     * Does not recursively list files. Uses delimiter-based listing for efficient implementation.
     * Returned directory names are sorted lexicographically. If the location does not exist, return an empty vector.
     *
     * @param prefix Returns directories from the given prefix only.
     * @return promise<std::vector<std::string>> Vector of directory paths (without trailing delimiter).
     */
    virtual async::promise<std::vector<std::string>> list_dirs(const std::string& prefix) const = 0;

    virtual async::promise<std::optional<resource_meta>> metadata(const std::string& path) const
    {
        throw reader_error{path, 501, "not_implemented"};
    }

    /**
     * @brief Downloads the resource, whole or slice of it, depending on the range.
     *
     * @param path The path of the resource.
     * @param range The range of the resource to download. [0, 0] downloads the whole resource.
     * @param options The options to use during fetching.
     * @return promise<base::read_buffer> The resource data in bytes at the specified range.
     */
    virtual async::promise<base::read_buffer>
    download(const std::string& path, std::pair<int64_t, int64_t> range, fetch_options options) const = 0;

    /**
     * Alternate version of download, which returns an optional instead of throwing an exception.
     * @param path The path of the resource.
     * @param range The range of the resource to download. [0, 0] downloads the whole resource.
     * @param options The options to use during fetching.
     * @return async::promise<std::optional<base::read_buffer>> The resource data in bytes at the specified range. Or
     * nullopt if the resource does not exist.
     */
    async::promise<std::optional<base::read_buffer>>
    download_optional(const std::string& path, std::pair<int64_t, int64_t> range, fetch_options options);

    /**
     * @brief Checks whether the resource exists.
     *
     * @param path The path of the resource to check.
     * @return promise<bool> True if the resource exists.
     */
    virtual async::promise<bool> exists(const std::string& path) const;

    /**
     * @brief Downloads the entire resource and parses it as a JSON object.
     *
     * @param path The path of the resource.
     * @param options The options to use during fetching.
     * @return promise<icm::const_json> The JSON data object.
     * @return int Unique id of the request.
     */
    async::promise<icm::const_json> download_json(const std::string& path, fetch_options options = fetch_options()) const;

    /**
     * @brief Downloads the entire resource and parses it as a JSONL object.
     *
     * @param path The path of the resource.
     * @param options The options to use during fetching.
     * @return promise<std::vector<icm::const_json>> The JSON objects.
     */
    async::promise<std::vector<icm::const_json>> download_jsonl(const std::string& path,
                                                                fetch_options options = fetch_options()) const;

    /**
     * @brief Returns reader for the given subpath of this reader.
     *
     * @param subpath Subpath.
     * @return std::shared_ptr<reader> New reader.
     */
    virtual std::shared_ptr<reader> reader_for_subpath(const std::string& subpath) const = 0;

    /**
     * @brief pre-sign url with the reader credentials
     * @param path file path in blob
     * @return promise<std::string> string promise or empty string if signing failes.
     */
    virtual async::promise<std::string> sign_url(const std::string& path) const = 0;

public:
    /**
     * @brief get the writer
     * @info this is needed in cases if we want to materialize the dataset views
     * @note this function can throw an exception if the reader does not support write
     *  this can be in case it the user does not have write permissions
     * @return std::shared_ptr<writer>
     */
    virtual std::shared_ptr<writer> get_writer() = 0;

    virtual ~reader() noexcept = default;

    static inline std::string range_to_header(std::pair<int64_t, int64_t> range)
    {
        if (range.second == 0) {
            return std::string("bytes=") + std::to_string(range.first) + std::string("-");
        }
        return std::string("bytes=") + std::to_string(range.first) + std::string("-") +
               std::to_string(range.second - 1);
    }
};

} // namespace storage
