#pragma once

/**
 * @file async_prefetcher.hpp
 * @brief Definition of the `async_prefetcher` class.
 */

#include <async/promise.hpp>
#include <deeplake_core/sample.hpp>
#include <heimdall/dataset_view.hpp>
#include <icm/string_set.hpp>

#include <memory>

namespace bifrost {

/**
 * @brief Class to prefetch samples from a dataset.
 * This class is used to prefetch samples from a dataset and provide them to the user. It is designed to be used in a
 * multi-threaded environment.
 */
class async_prefetcher
{
public:
    async_prefetcher(heimdall::dataset_view_ptr ds,
                     bool drop_last,
                     std::optional<int64_t> batch_size = std::nullopt,
                     icm::string_set raw_columns = {},
                     bool ignore_errors = false);

    ~async_prefetcher();

    async_prefetcher(const async_prefetcher&) = delete;
    async_prefetcher& operator=(const async_prefetcher&) = delete;
    async_prefetcher(async_prefetcher&&) noexcept = default;
    async_prefetcher& operator=(async_prefetcher&&) noexcept = default;

    void start();
    void stop() noexcept;

    /**
     * @brief Wait for the first batch to be ready (for cold run optimization).
     * @param timeout_ms Maximum time to wait in milliseconds.
     *
     * This method is used for eager prefetching during cold runs.
     * It fetches and caches the first batch so that subsequent next_batch()
     * calls return immediately without blocking.
     */
    void wait_for_first_batch(int64_t timeout_ms = 30000);

    bool is_started() const noexcept;

    heimdall::dataset_view_ptr dataset() const noexcept;

    int64_t batch_size() const noexcept;

    int64_t size() const noexcept;

public:
    deeplake_core::batch next_batch();

    // Add async version of next_batch
    async::promise<deeplake_core::batch> next_batch_async();

private:
    class impl;
    std::shared_ptr<impl> impl_;
};

} // namespace bifrost
