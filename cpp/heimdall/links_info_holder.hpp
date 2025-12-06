#pragma once

/**
 * @file links_info_holder.hpp
 * @brief Definition of the links_info_holder interface.
 */

#include <storage/fetch_options.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace async {
template <typename T>
class promise;
}

namespace nd {
class array;
}

namespace heimdall {

class column_view;

class links_info_holder
{
public:
    explicit links_info_holder(std::shared_ptr<const column_view> source);

    virtual ~links_info_holder() = default;

    /// @name Links info fetching.
    /// @{

    /**
     * @brief Request links info for a single sample.
     * @return A promise that will be fulfilled with the links info.
     */
    virtual async::promise<std::string> creds_key() const;

    /**
     * @brief Request links info for a single sample.
     * @param index The index of the sample.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the links info.
     */
    virtual async::promise<nd::array> request_links_info(int64_t index, storage::fetch_options options) const;

    /**
     * @brief Request links info for a range of samples.
     * @param start_index The start index of the range.
     * @param end_index The end index of the range.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the sample info.
     */
    virtual async::promise<nd::array>
    request_links_info_range(int64_t start_index, int64_t end_index, storage::fetch_options options) const;

    /**
     * @brief Request links info for all samples.
     * @param options The fetch options.
     * @return A promise that will be fulfilled with the links info.
     */
    virtual async::promise<nd::array> request_links_info_full(storage::fetch_options options) const;
    /// @}

private:
    std::shared_ptr<const heimdall::column_view> source_;
};

} // namespace heimdall
