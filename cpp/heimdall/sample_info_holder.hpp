#pragma once

/**
 * @file sample_info_holder.hpp
 * @brief Definition of the sample_info_holder interface.
 */

#include <storage/fetch_options.hpp>

#include <icm/const_json.hpp>

#include <cstdint>
#include <vector>

namespace async {
template <typename T>
class promise;
}

namespace heimdall {

class sample_info_holder
{
public:
    virtual ~sample_info_holder() = default;

public:
    /// @name Sample info fetching.
    /// @{

    /**
     * @brief Request sample info for a single sample.
     * @param index The index of the sample.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the sample info.
     */
    virtual async::promise<icm::const_json> request_sample_info(int64_t index, storage::fetch_options options) = 0;

    /**
     * @brief Request sample info for a range of samples.
     * @param start_index The start index of the range.
     * @param end_index The end index of the range.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the sample info.
     */
    virtual async::promise<std::vector<icm::const_json>>
    request_sample_info_range(int64_t start_index, int64_t end_index, storage::fetch_options options) = 0;

    /**
     * @brief Request sample info for all samples.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the sample info.
     */
    virtual async::promise<std::vector<icm::const_json>>
    request_sample_info_full(storage::fetch_options options) = 0;
    /// @}

public:
    /**
     * @brief Request sample info for multiple samples.
     * @param indices The indices of the samples.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the sample info.
     */
    async::promise<std::vector<icm::const_json>> request_multiple_sample_info(const std::vector<int64_t>& indices,
                                                                                 storage::fetch_options options);
};

} // namespace heimdall
