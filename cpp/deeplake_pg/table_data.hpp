#pragma once

#include "nd_utils.hpp"
#include "utils.hpp"

#include <base/spin_lock.hpp>
#include <bifrost/column_streamer.hpp>
#include <deeplake_api/dataset.hpp>
#include <icm/vector.hpp>
#include <nd/array.hpp>
#include <nd/string_array_holder.hpp>

#include <fmt/format.h>
#include <http/uri.hpp>

#include <access/htup.h>
#include <access/tupdesc.h>
#include <miscadmin.h>
#include <postgres.h>
#include <utils/date.h>
#include <utils/numeric.h>
#include <utils/rel.h>
#include <utils/timestamp.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace pg {

namespace impl {
using string_stream_array_holder = nd::string_stream_array_holder;
} // namespace impl

struct table_data
{
    inline table_data(Oid table_oid,
                      const std::string& table_name,
                      TupleDesc tupdesc,
                      std::string dataset_path,
                      icm::string_map<> creds);
    table_data(const table_data&) = delete;
    table_data& operator=(const table_data&) = delete;
    inline table_data(table_data&& o) = default;
    inline table_data& operator=(table_data&& o) = default;
    ~table_data()
    {
        if (refresh_promise_ && !refresh_promise_.is_cancelled()) {
            refresh_promise_.get_future().get();
        }
    }

    inline void commit(bool show_progress = false);
    inline void open_dataset(bool create = false);
    inline void refresh();
    inline const std::string& get_table_name() const noexcept;
    inline const std::shared_ptr<deeplake_api::dataset>& get_dataset() const noexcept;
    inline const std::shared_ptr<deeplake_api::dataset>& get_read_only_dataset() const noexcept;
    inline const http::uri& get_dataset_path() const noexcept;
    inline heimdall::column_view_ptr get_column_view(int32_t column_idx) const;
    inline TupleDesc get_tuple_descriptor() const noexcept;
    inline Oid get_atttypid(AttrNumber attr_num) const noexcept;
    inline int32_t get_atttypmod(AttrNumber attr_num) const noexcept;
    inline Oid get_base_atttypid(AttrNumber attr_num) const noexcept;
    inline int32_t get_attndims(AttrNumber attr_num) const noexcept;
    inline std::string get_atttypename(AttrNumber attr_num) const noexcept;
    inline bool is_column_dropped(AttrNumber attr_num) const noexcept;
    inline int32_t get_tupdesc_index(AttrNumber attr_num) const noexcept;
    inline bool is_column_nullable(AttrNumber attr_num) const noexcept;
    inline bool is_column_indexed(AttrNumber attr_num) const noexcept;
    inline int32_t num_columns() const noexcept;
    inline int64_t num_rows() const noexcept;
    inline int64_t num_total_rows() const noexcept;
    inline void reset_insert_rows() noexcept;
    inline void add_insert_slots(int32_t nslots, TupleTableSlot** slots);
    inline void add_delete_row(int64_t row_id);
    inline void clear_delete_rows() noexcept;
    inline void add_update_row(int64_t row_id, icm::string_map<nd::array> update_row);
    inline void clear_update_rows() noexcept;
    inline Oid get_table_oid() const noexcept;
    inline bool flush();

private:
    bool flush_inserts(bool full_flush = false);
    bool flush_deletes();
    bool flush_updates();
    inline void force_refresh();

public:
    /// @name Streamer management
    /// @{
    struct streamer_info
    {
        struct batch_data
        {
            std::atomic<bool> initialized_{false};
            nd::array owner_;
            const uint8_t* data_ = nullptr;
            impl::string_stream_array_holder holder_;

            batch_data() = default;
            batch_data(const batch_data&) = delete;
            batch_data(batch_data&& other) noexcept
                : initialized_(other.initialized_.load())
                , owner_(std::move(other.owner_))
                , data_(other.data_)
                , holder_(std::move(other.holder_))
            {
                other.data_ = nullptr;
            }
            batch_data& operator=(const batch_data&) = delete;
            batch_data& operator=(batch_data&&) = delete;
        };

        struct column_data
        {
            std::mutex mutex_;
            std::vector<batch_data> batches;
            /// Pre-fetched raw batches from parallel prefetch. Protected by mutex_.
            /// warm_all_streamers() drains the streamer into this deque;
            /// value_ptr/get_sample/value consume from the front.
            std::deque<nd::array> prefetched_raw_batches_;

            column_data() = default;
            column_data(const column_data&) = delete;
            column_data(column_data&& other) noexcept
                : batches(std::move(other.batches))
                , prefetched_raw_batches_(std::move(other.prefetched_raw_batches_))
            {
            }
            column_data& operator=(const column_data&) = delete;
            column_data& operator=(column_data&&) = delete;
        };

        std::vector<column_data> column_to_batches;
        std::vector<std::unique_ptr<bifrost::column_streamer>> streamers;

        inline void reset() noexcept
        {
            column_to_batches.clear();
            streamers.clear();
        }

        /**
         * @brief Pre-warm all streamers by triggering parallel first batch downloads.
         *
         * This method initiates the download of the first batch for all active
         * streamers in parallel, then waits for all downloads to complete.
         * This significantly improves cold run performance by overlapping the
         * initial data fetches.
         */
        inline void warm_all_streamers();


        /**
         * @brief Pre-initialize batches for all given columns at the specified row in parallel.
         *
         * For a cold run, batch initialization blocks on I/O. Without this method,
         * the scan processes columns sequentially, serializing I/O waits.
         * This method triggers batch downloads for all columns that need it concurrently,
         * then waits for all to complete, so the subsequent sequential column processing
         * finds all batches already initialized.
         */
        inline void prefetch_batches_for_row(const std::vector<int32_t>& column_indices, int64_t row_number);

        inline nd::array get_sample(int32_t column_number, int64_t row_number);

        template <typename T>
        inline T value(int32_t column_number, int64_t row_number);

        template <typename T>
        inline const T* value_ptr(int32_t column_number, int64_t row_number);
    };

    inline streamer_info& get_streamers() noexcept;
    inline std::pair<int64_t, int64_t> get_row_range(int32_t worker_id) const;
    inline void create_streamer(int32_t idx, int32_t worker_id);
    inline bool column_has_streamer(uint32_t idx) const noexcept;
    inline void reset_streamers() noexcept;
    inline nd::array get_column_value(int32_t column_number, int64_t row_number) const noexcept;
    inline nd::array get_sample(int32_t column_number, int64_t row_number);
    inline bool is_column_requested(int32_t column_number) const noexcept;
    inline void set_column_requested(int32_t column_number, bool requested) noexcept;
    inline void reset_requested_columns() noexcept;
    inline bool is_star_selected() const noexcept;
    inline bool can_stream_column(int32_t column_number) const noexcept;
    /// @}

private:
    constexpr static size_t batch_size_ = 1u << 16; // Default batch size for tensor streamer
    constexpr static int32_t batch_size_log2_ = std::countr_zero(batch_size_);
    constexpr static int64_t batch_mask_ = batch_size_ - 1;

    streamer_info streamers_;
    icm::string_map<icm::vector<nd::array>> insert_rows_;
    std::deque<async::promise<void>> insert_promises_;
    icm::vector<int64_t> delete_rows_;
    icm::vector<std::tuple<int64_t, std::string, nd::array>> update_rows_;
    std::shared_ptr<deeplake_api::dataset> dataset_;
    std::shared_ptr<deeplake_api::dataset> refreshing_dataset_;
    async::promise<void> refresh_promise_;
    icm::vector<bool> requested_columns_;
    icm::vector<Oid> base_typeids_;              // Cached base type OIDs for performance
    icm::vector<int32_t> active_column_indices_; // Maps logical index to TupleDesc index (excludes dropped)
    icm::string_map<> creds_;
    TupleDesc tuple_descriptor_;
    http::uri dataset_path_ = http::uri(std::string());
    std::string table_name_;
    Oid table_oid_ = InvalidOid;
    int64_t num_total_rows_ = 0;
    uint64_t cached_version_ = 0; // Cached version from shared memory for detecting changes
    bool is_star_selected_ = true;
};

} // namespace pg

// Include inline implementation functions
#include "table_data_impl.hpp"
