#pragma once

// NOTE: postgres.h and libintl.h must be included first in the compilation unit before this header

#include "memory_tracker.hpp"
#include "progress_utils.hpp"
#include "table_version.hpp"
#include "utils.hpp"

#include <heimdall_common/filtered_column.hpp>

// Inline implementation functions for table_data
// This file should be included at the end of table_data.hpp

namespace pg {

namespace impl {

inline async::promise<void>
append_rows(std::shared_ptr<deeplake_api::dataset> dataset, icm::string_map<nd::array> rows, int32_t num_rows)
{
    ASSERT(num_rows > 0);

    return async::run_on_main([ds = std::move(dataset), rows = std::move(rows), num_rows]() mutable {
        if (num_rows == 1) {
            return ds->append_row(std::move(rows));
        }
        return ds->append_rows(std::move(rows));
    });
}

inline void commit_dataset(std::shared_ptr<deeplake_api::dataset> dataset, bool show_progress)
{
    constexpr auto high_num_rows = 50000;
    const bool print_progress =
        (show_progress && dataset->num_rows() > high_num_rows && dataset->has_uncommitted_changes());
    auto promise = async::run_on_main([ds = std::move(dataset)]() {
        return ds->commit();
    });
    if (print_progress) {
        const std::string message = fmt::format("Committing dataset (samples: {})", dataset->num_rows());
        pg::utils::print_progress_and_wait(std::move(promise), message);
    } else {
        promise.get_future().get();
    }
}

} // namespace impl

inline table_data::table_data(
    Oid table_oid, const std::string& table_name, TupleDesc tupdesc, std::string dataset_path, icm::string_map<> creds)
    : table_name_(table_name)
    , table_oid_(table_oid)
    , tuple_descriptor_(tupdesc)
    , dataset_path_(std::move(dataset_path))
    , creds_(std::move(creds))
{
    // Build list of active (non-dropped) column indices
    // This maps logical index (0, 1, 2, ...) to TupleDesc index
    for (int32_t i = 0; i < tuple_descriptor_->natts; ++i) {
        Form_pg_attribute attr = TupleDescAttr(tuple_descriptor_, i);
        if (!attr->attisdropped) {
            active_column_indices_.push_back(i);
        }
    }

    const auto num_active = active_column_indices_.size();
    requested_columns_.resize(num_active, false);
    base_typeids_.resize(num_active);

    // Cache base type OIDs for active columns only
    for (size_t i = 0; i < num_active; ++i) {
        const auto tupdesc_idx = active_column_indices_[i];
        Form_pg_attribute attr = TupleDescAttr(tuple_descriptor_, tupdesc_idx);
        base_typeids_[i] = pg::utils::get_base_type(attr->atttypid);
    }
}

inline void table_data::commit(bool show_progress)
{
    if (dataset_ == nullptr || !dataset_->has_uncommitted_changes()) {
        return;
    }
    try {
        flush();
        impl::commit_dataset(get_dataset(), show_progress);
    } catch (const std::exception& e) {
        reset_insert_rows();
        clear_delete_rows();
        clear_update_rows();
        elog(ERROR, "Failed to commit dataset: %s", e.what());
    } catch (...) {
        reset_insert_rows();
        clear_delete_rows();
        clear_update_rows();
        elog(ERROR, "Failed to commit dataset, unknown exception");
    }
    streamers_.reset();
    force_refresh();
}

inline void table_data::open_dataset(bool create)
{
    elog(DEBUG1, "Opening dataset at path: %s (create=%s)", dataset_path_.url().c_str(), create ? "true" : "false");
    try {
        auto creds = creds_;
        if (create) {
            dataset_ = deeplake_api::create(dataset_path_, std::move(creds)).get_future().get();
        } else {
            dataset_ = deeplake_api::open(dataset_path_, std::move(creds)).get_future().get();
        }
        ASSERT(dataset_ != nullptr);
        num_total_rows_ = dataset_->num_rows();

        // Enable logging if GUC parameter is set
        if (pg::enable_dataset_logging && dataset_ && !dataset_->is_logging_enabled()) {
            dataset_->start_logging();
            elog(DEBUG1, "Dataset logging enabled for: %s", table_name_.c_str());
        }

        if (!pg::use_shared_mem_for_refresh) {
            refreshing_dataset_ = dataset_->clone();
            ASSERT(refreshing_dataset_ != nullptr);
            refreshing_dataset_->set_indexing_mode(deeplake::indexing_mode::off);
        }
    } catch (const std::exception& e) {
        auto s = create ? "create" : "open";
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to %s table storage: %s", s, e.what()),
                 errdetail("Table name: %s, Dataset path: %s", table_name_.c_str(), dataset_path_.url().c_str()),
                 errhint("Check if the dataset path is accessible and has proper permissions")));
    }
}

inline void table_data::refresh()
{
    if (!dataset_ || dataset_->has_uncommitted_changes()) {
        return;
    }
    if (pg::use_shared_mem_for_refresh) {
        // Check if table version has changed and trigger refresh if needed
        const uint64_t current_version = table_version_tracker::get_version(table_oid_);
        if (current_version != cached_version_) {
            cached_version_ = current_version;
            streamers_.reset();
            async::run_on_main([this]() {
                return dataset_->refresh();
            })
                .get_future()
                .get();
            // After refresh from version change (another backend committed),
            // use the dataset's actual row count. This correctly handles both
            // inserts and deletes from other backends.
            num_total_rows_ = dataset_->num_rows();
        }
    } else {
        ASSERT(refreshing_dataset_ != nullptr);
        if (refresh_promise_ && !refresh_promise_.is_ready()) {
            return; // Already refreshing
        } else if (refresh_promise_ && refresh_promise_.is_ready()) {
            std::move(refresh_promise_).get();
            if (refreshing_dataset_->version() != dataset_->version()) { // refresh happend!
                streamers_.reset();
                const auto ds_indexing_mode = dataset_->get_indexing_mode();
                std::swap(dataset_, refreshing_dataset_);
                dataset_->set_indexing_mode(ds_indexing_mode);
                refreshing_dataset_->set_indexing_mode(deeplake::indexing_mode::off);
                // After refresh from version change (another backend committed),
                // use the dataset's actual row count. This correctly handles both
                // inserts and deletes from other backends.
                num_total_rows_ = dataset_->num_rows();
            }
        }
        // trigger new refresh
        refresh_promise_ = async::run_on_main([d = refreshing_dataset_]() {
            return d->refresh();
        });
    }
}

inline void table_data::force_refresh()
{
    if (pg::use_shared_mem_for_refresh) {
        return;
    }
    if (refresh_promise_ && !refresh_promise_.cancel()) {
        refresh_promise_.get_future().get();
    }
    ASSERT(refreshing_dataset_ != nullptr);
    refresh_promise_ = async::run_on_main([d = refreshing_dataset_]() {
        return d->refresh();
    });
}

inline const std::string& table_data::get_table_name() const noexcept
{
    return table_name_;
}

inline const std::shared_ptr<deeplake_api::dataset>& table_data::get_dataset() const noexcept
{
    if (dataset_ == nullptr) {
        const_cast<table_data*>(this)->open_dataset();
    }
    return dataset_;
}

inline const std::shared_ptr<deeplake_api::dataset>& table_data::get_read_only_dataset() const noexcept
{
    return get_dataset();
}

inline const http::uri& table_data::get_dataset_path() const noexcept
{
    return dataset_path_;
}

inline heimdall::column_view_ptr table_data::get_column_view(int32_t column_idx) const
{
    return (*get_read_only_dataset())[column_idx].shared_from_this();
}

inline TupleDesc table_data::get_tuple_descriptor() const noexcept
{
    return tuple_descriptor_;
}

inline Oid table_data::get_atttypid(AttrNumber attr_num) const noexcept
{
    return get_base_atttypid(attr_num);
}

inline int32_t table_data::get_atttypmod(AttrNumber attr_num) const noexcept
{
    const auto tupdesc_idx = active_column_indices_[attr_num];
    return TupleDescAttr(tuple_descriptor_, tupdesc_idx)->atttypmod;
}

inline Oid table_data::get_base_atttypid(AttrNumber attr_num) const noexcept
{
    return base_typeids_[attr_num];
}

inline int32_t table_data::get_attndims(AttrNumber attr_num) const noexcept
{
    const auto tupdesc_idx = active_column_indices_[attr_num];
    return TupleDescAttr(tuple_descriptor_, tupdesc_idx)->attndims;
}

inline std::string table_data::get_atttypename(AttrNumber attr_num) const noexcept
{
    const auto tupdesc_idx = active_column_indices_[attr_num];
    return NameStr(TupleDescAttr(tuple_descriptor_, tupdesc_idx)->attname);
}

inline bool table_data::is_column_dropped(AttrNumber attr_num) const noexcept
{
    // Active columns are never dropped (dropped columns are filtered out)
    (void)attr_num;
    return false;
}

inline bool table_data::is_column_nullable(AttrNumber attr_num) const noexcept
{
    const auto tupdesc_idx = active_column_indices_[attr_num];
    return TupleDescAttr(tuple_descriptor_, tupdesc_idx)->attnotnull ? false : true;
}

inline int32_t table_data::get_tupdesc_index(AttrNumber attr_num) const noexcept
{
    return active_column_indices_[attr_num];
}

inline bool table_data::is_column_indexed(AttrNumber attr_num) const noexcept
{
    return pg::pg_index::get_oid(table_name_, get_atttypename(attr_num)) != InvalidOid;
}

inline int32_t table_data::num_columns() const noexcept
{
    return static_cast<int32_t>(active_column_indices_.size());
}

inline int64_t table_data::num_rows() const noexcept
{
    return get_read_only_dataset()->num_rows();
}

inline int64_t table_data::num_total_rows() const noexcept
{
    // If num_total_rows_ is 0, the dataset may not have been opened yet
    // (e.g., when table_data was loaded from metadata after RENAME COLUMN).
    // In this case, open the dataset to get the correct row count.
    if (num_total_rows_ == 0 && dataset_ == nullptr) {
        const_cast<table_data*>(this)->open_dataset();
    }
    return num_total_rows_;
}

inline void table_data::reset_insert_rows() noexcept
{
    for (auto& p : insert_promises_) {
        p.cancel();
    }
    insert_promises_.clear();
    if (dataset_) {
        num_total_rows_ = dataset_->num_rows();
    }
}

inline void table_data::add_insert_slots(int32_t nslots, TupleTableSlot** slots)
{
    for (int32_t k = 0; k < nslots; ++k) {
        auto slot = slots[k];
        slot_getallattrs(slot);
    }
    for (int32_t i = 0; i < num_columns(); ++i) {
        auto& column_values = insert_rows_[get_atttypename(i)];
        const auto dt = get_column_view(i)->dtype();
        for (int32_t k = 0; k < nslots; ++k) {
            auto slot = slots[k];
            nd::array val;
            if (slot->tts_isnull[i]) {
                val = (nd::dtype_is_numeric(dt) ? nd::adapt(0) : nd::none(dt, 0));
            } else {
                val = pg::utils::datum_to_nd(slot->tts_values[i], get_base_atttypid(i), get_atttypmod(i));
            }
            column_values.push_back(std::move(val));
        }
    }
    num_total_rows_ += nslots;
    const auto num_inserts = insert_rows_.begin()->second.size();
    if (num_inserts >= 512) {
        flush_inserts();
    }
}

inline void table_data::add_delete_row(int64_t row_id)
{
    delete_rows_.push_back(row_id);
}

inline void table_data::clear_delete_rows() noexcept
{
    delete_rows_.clear();
}

inline void table_data::add_update_row(int64_t row_id, icm::string_map<nd::array> update_row)
{
    for (auto& [column_name, new_value] : update_row) {
        if (new_value.is_none() && new_value.dtype() == nd::dtype::unknown) {
            continue; // Skip updates with None values
        }
        update_rows_.emplace_back(row_id, std::move(column_name), std::move(new_value));
    }
}

inline void table_data::clear_update_rows() noexcept
{
    update_rows_.clear();
}

inline table_data::streamer_info& table_data::get_streamers() noexcept
{
    return streamers_;
}

inline bool table_data::column_has_streamer(uint32_t idx) const noexcept
{
    return streamers_.streamers.size() > idx && streamers_.streamers[idx] != nullptr;
}

inline void table_data::reset_streamers() noexcept
{
    streamers_.reset();
}

inline nd::array table_data::get_column_value(int32_t column_number, int64_t row_number) const noexcept
{
    return async::run_on_main([cv = get_column_view(column_number), row_number]() {
               return cv->request_sample(row_number, {});
           })
        .get_future()
        .get();
}

inline nd::array table_data::get_sample(int32_t column_number, int64_t row_number)
{
    if (!column_has_streamer(column_number)) {
        return get_column_value(column_number, row_number);
    }
    return streamers_.get_sample(column_number, row_number);
}

inline bool table_data::is_column_requested(int32_t column_number) const noexcept
{
    return requested_columns_[column_number];
}

inline void table_data::set_column_requested(int32_t column_number, bool requested) noexcept
{
    is_star_selected_ = false;
    requested_columns_[column_number] = requested;
}

inline void table_data::reset_requested_columns() noexcept
{
    is_star_selected_ = true;
    requested_columns_.assign(num_columns(), false);
}

inline bool table_data::is_star_selected() const noexcept
{
    return is_star_selected_;
}

inline bool table_data::can_stream_column(int32_t column_number) const noexcept
{
    if (column_number < 0 || column_number >= num_columns() || num_rows() == 0 ||
        type_is_array(get_atttypid(column_number))) {
        return false;
    }
    const auto column_width =
        pg::utils::get_column_width(get_base_atttypid(column_number), get_atttypmod(column_number));
    return column_width > 0 && column_width < pg::max_streamable_column_width;
}

inline bool table_data::flush_inserts(bool full_flush)
{
    if (!insert_rows_.empty()) {
        icm::string_map<nd::array> deeplake_rows;
        const auto num_inserts = insert_rows_.begin()->second.size();
        if (num_inserts == 1) {
            for (auto& [column_name, values] : insert_rows_) {
                deeplake_rows[column_name] = std::move(values.front());
            }
        } else {
            for (auto& [column_name, values] : insert_rows_) {
                deeplake_rows[column_name] = nd::dynamic(std::move(values));
            }
        }
        insert_rows_.clear();
        streamers_.reset();
        insert_promises_.push_back(impl::append_rows(get_dataset(), std::move(deeplake_rows), num_inserts));
    }
    try {
        constexpr size_t max_pending_insert_promises = 1024;
        while (!insert_promises_.empty() && (full_flush || insert_promises_.size() >= max_pending_insert_promises)) {
            auto p = std::move(insert_promises_.front());
            insert_promises_.pop_front();
            if (p.is_ready()) {
                std::move(p).get();
            } else {
                p.get_future().get();
            }
        }
    } catch (const base::exception& e) {
        elog(WARNING, "Failed to flush inserts: %s", e.what());
        reset_insert_rows();
        return false;
    }

    return true;
}

inline bool table_data::flush_deletes()
{
    if (delete_rows_.empty()) {
        return true;
    }

    const auto num_deletes = static_cast<int64_t>(delete_rows_.size());

    // Flush the delete rows to the dataset
    try {
        streamers_.reset();
        get_dataset()->delete_rows(delete_rows_);
    } catch (const base::exception& e) {
        elog(WARNING, "Failed to flush deletes: %s", e.what());
        delete_rows_.clear();
        return false;
    }

    delete_rows_.clear();

    // Update the total row count to reflect deleted rows
    num_total_rows_ -= num_deletes;

    return true;
}

inline bool table_data::flush_updates()
{
    if (update_rows_.empty()) {
        return true;
    }

    // Flush the update rows to the dataset
    try {
        streamers_.reset();
        icm::vector<async::promise<void>> update_promises;
        update_promises.reserve(update_rows_.size());
        for (const auto& [row_number, column_name, new_value] : update_rows_) {
            update_promises.emplace_back(get_dataset()->update_row(row_number, column_name, new_value));
        }
        async::combine(std::move(update_promises)).get_future().get();
    } catch (const base::exception& e) {
        elog(WARNING, "Failed to flush updates: %s", e.what());
        update_rows_.clear();
        return false;
    }

    update_rows_.clear();
    return true;
}

inline Oid table_data::get_table_oid() const noexcept
{
    return table_oid_;
}

inline std::pair<int64_t, int64_t> table_data::get_row_range(int32_t worker_id) const
{
    ASSERT(worker_id >= 0 && worker_id < max_parallel_workers);
    const auto total_rows = num_total_rows();
    const auto total_workers = max_parallel_workers;
    int64_t rows_per_worker = total_rows / total_workers;
    int64_t remaining_rows = total_rows % total_workers;
    auto start_row = worker_id * rows_per_worker;
    auto end_row = start_row + rows_per_worker;
    if (worker_id == 0) {
        end_row += remaining_rows; // First worker gets the remaining rows
    }
    return {start_row, end_row};
}

inline void table_data::create_streamer(int32_t idx, int32_t worker_id)
{
    const auto col_count = num_columns();
    if (streamers_.streamers.empty()) {
        streamers_.streamers.resize(col_count);
        streamers_.column_to_batches.resize(col_count);
    }
    ASSERT(idx >= 0 && idx < col_count);
    if (streamers_.streamers[idx]) {
        return; // Already created
    }
    if (pg::memory_tracker::has_memory_limit()) {
        const auto column_size = pg::utils::get_column_width(get_base_atttypid(idx), get_atttypmod(idx)) * num_total_rows();
        pg::memory_tracker::ensure_memory_available(column_size);
    }
    heimdall::column_view_ptr cv = get_column_view(idx);
    if (worker_id != -1) {
        auto [start_row, end_row] = get_row_range(worker_id);
        cv = heimdall_common::create_filtered_column(*(cv),
                                                     icm::index_mapping_t<int64_t>::slice({start_row, end_row, 1}));
    }
    streamers_.streamers[idx] = std::make_unique<bifrost::column_streamer>(cv, batch_size_);
    const int64_t row_count = num_total_rows();
    const int64_t batch_count = (row_count + batch_size_ - 1) / batch_size_;
    streamers_.column_to_batches[idx].batches.resize(batch_count);
}

inline nd::array table_data::streamer_info::get_sample(int32_t column_number, int64_t row_number)
{
    const int64_t batch_index = row_number >> batch_size_log2_;
    const int64_t row_in_batch = row_number & batch_mask_;

    auto& col_data = column_to_batches[column_number];
    auto& batch = col_data.batches[batch_index];
    if (!batch.initialized_.load(std::memory_order_acquire)) [[unlikely]] {
        std::lock_guard lock(col_data.mutex_);
        for (int64_t i = 0; i <= batch_index; ++i) {
            if (!col_data.batches[i].initialized_.load(std::memory_order_relaxed)) {
                col_data.batches[i].owner_ = streamers[column_number]->next_batch();
                col_data.batches[i].initialized_.store(true, std::memory_order_release);
            }
        }
    }
    return batch.owner_[static_cast<size_t>(row_in_batch)];
}

template <typename T>
inline T table_data::streamer_info::value(int32_t column_number, int64_t row_number)
{
    return *(value_ptr<T>(column_number, row_number));
}

template <typename T>
inline const T* table_data::streamer_info::value_ptr(int32_t column_number, int64_t row_number)
{
    const int64_t batch_index = row_number >> batch_size_log2_;
    const int64_t row_in_batch = row_number & batch_mask_;

    auto& col_data = column_to_batches[column_number];
    auto& batch = col_data.batches[batch_index];
    if (!batch.initialized_.load(std::memory_order_acquire)) [[unlikely]] {
        std::lock_guard lock(col_data.mutex_);
        for (int64_t i = 0; i <= batch_index; ++i) {
            if (!col_data.batches[i].initialized_.load(std::memory_order_relaxed)) {
                col_data.batches[i].owner_ = utils::eval_with_nones<T>(streamers[column_number]->next_batch());
                col_data.batches[i].data_ = col_data.batches[i].owner_.data().data();
                col_data.batches[i].initialized_.store(true, std::memory_order_release);
            }
        }
    }

    return reinterpret_cast<const T*>(batch.data_) + row_in_batch;
}

template <>
inline std::string_view table_data::streamer_info::value(int32_t column_number, int64_t row_number)
{
    const int64_t batch_index = row_number >> batch_size_log2_;
    const int64_t row_in_batch = row_number & batch_mask_;

    auto& col_data = column_to_batches[column_number];
    auto& batch = col_data.batches[batch_index];
    if (!batch.initialized_.load(std::memory_order_acquire)) [[unlikely]] {
        std::lock_guard lock(col_data.mutex_);
        for (int64_t i = 0; i <= batch_index; ++i) {
            if (!col_data.batches[i].initialized_.load(std::memory_order_relaxed)) {
                col_data.batches[i].owner_ = streamers[column_number]->next_batch();
                col_data.batches[i].holder_ = impl::string_stream_array_holder(col_data.batches[i].owner_);
                col_data.batches[i].initialized_.store(true, std::memory_order_release);
            }
        }
    }

    return batch.holder_.data(static_cast<size_t>(row_in_batch));
}

inline bool table_data::flush()
{
    const bool s1 = flush_inserts(true);
    const bool s2 = flush_deletes();
    const bool s3 = flush_updates();
    return s1 && s2 && s3;
}

} // namespace pg
