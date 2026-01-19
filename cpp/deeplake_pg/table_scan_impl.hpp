#pragma once

#include "table_data.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

namespace pg::impl {

// Helper function to check how column should be processed
inline table_scan::column_process_result
should_process_column(table_data& table_data, int32_t column_index, bool only_streamer) noexcept
{
    if (!table_data.is_column_requested(column_index)) {
        if (pg::query_info::current().has_current_score() && table_data.is_column_indexed(column_index)) [[unlikely]] {
            return table_scan::column_process_result::skip_as_scored; // Set to false/not null for indexed columns with
                                                                      // score
        } else {
            return table_scan::column_process_result::skip_as_null;
        }
    }

    if (only_streamer && !table_data.column_has_streamer(column_index)) {
        return table_scan::column_process_result::skip_as_special;
    }

    return table_scan::column_process_result::process;
}

} // namespace pg::impl

namespace pg {

inline table_scan::table_scan(Oid table_id, bool is_parallel, bool streamer_only)
    : table_data_(table_storage::instance().get_table_data(table_id))
    , current_position_(0)
    , num_rows_(table_data_.num_total_rows())
    , table_id_(table_id)
    , is_parallel_(is_parallel)
{
    for (int32_t i = 0; i < table_data_.num_columns(); ++i) {
        auto process_result = impl::should_process_column(table_data_, i, streamer_only);
        if (process_result == table_scan::column_process_result::process) {
            process_columns_.emplace_back(i);
        } else if (process_result == table_scan::column_process_result::skip_as_null) {
            null_columns_.emplace_back(i);
        } else if (process_result == table_scan::column_process_result::skip_as_scored) {
            scored_columns_.emplace_back(i);
        } else if (process_result == table_scan::column_process_result::skip_as_special) {
            special_columns_.emplace_back(i);
        }
    }
    if (is_parallel_ || IsParallelWorker()) {
        int32_t worker_number = ParallelWorkerNumber + 1;
        auto [start_row, end_row] = table_data_.get_row_range(worker_number);
        current_position_ = start_row;
        num_rows_ = end_row;
    }
}

inline std::pair<Datum, bool> table_scan::get_datum(int32_t column_number, int64_t row_number) const noexcept
{
    const auto base_typeid = table_data_.get_base_atttypid(column_number);
    const auto column_typmod = table_data_.get_atttypmod(column_number);
    if (!table_data_.column_has_streamer(column_number)) {
        auto res = table_data_.get_column_value(column_number, row_number);
        if (res.is_none()) {
            return {(Datum)0, true};
        }
        auto datum = pg::utils::nd_to_datum(std::move(res), base_typeid, column_typmod);
        return {datum, datum == (Datum)0};
    }
    switch (base_typeid) {
    case BOOLOID: {
        return {BoolGetDatum(table_data_.get_streamers().value<bool>(column_number, row_number)), false};
    }
    case INT2OID: {
        return {Int16GetDatum(table_data_.get_streamers().value<int16_t>(column_number, row_number)), false};
    }
    case INT4OID: {
        return {Int32GetDatum(table_data_.get_streamers().value<int32_t>(column_number, row_number)), false};
    }
    case INT8OID: {
        return {Int64GetDatum(table_data_.get_streamers().value<int64_t>(column_number, row_number)), false};
    }
    case FLOAT4OID: {
        return {Float4GetDatum(table_data_.get_streamers().value<float>(column_number, row_number)), false};
    }
    case FLOAT8OID: {
        return {Float8GetDatum(table_data_.get_streamers().value<double>(column_number, row_number)), false};
    }
    case NUMERICOID: {
        const auto val = table_data_.get_streamers().value<double>(column_number, row_number);
        return {DirectFunctionCall1(float8_numeric, Float8GetDatum(val)), false};
    }
    case DATEOID: {
        int32_t date = table_data_.get_streamers().value<int32_t>(column_number, row_number);
        date -= pg::POSTGRES_EPOCH_DAYS;
        return {DateADTGetDatum(date), false};
    }
    case TIMEOID: {
        int64_t time = table_data_.get_streamers().value<int64_t>(column_number, row_number);
        return {TimeADTGetDatum(static_cast<TimeADT>(time)), false};
    }
    case TIMESTAMPOID: {
        int64_t timestamp = table_data_.get_streamers().value<int64_t>(column_number, row_number);
        timestamp -= pg::TIMESTAMP_EPOCH_DIFF_US;
        return {TimestampGetDatum(static_cast<Timestamp>(timestamp)), false};
    }
    case TIMESTAMPTZOID: {
        int64_t timestamp_tz = table_data_.get_streamers().value<int64_t>(column_number, row_number);
        timestamp_tz -= pg::TIMESTAMP_EPOCH_DIFF_US;
        return {TimestampTzGetDatum(static_cast<TimestampTz>(timestamp_tz)), false};
    }
    case UUIDOID: {
        auto str = table_data_.get_streamers().value<std::string_view>(column_number, row_number);
        // Treat empty string as NULL for UUID columns (same as duckdb_deeplake_scan.cpp)
        if (str.empty()) {
            return {(Datum)0, true};
        }
        std::string str_copy(str.data(), str.size());
        Datum uuid = DirectFunctionCall1(uuid_in, CStringGetDatum(str_copy.c_str()));
        return {uuid, false};
    }
    case CHAROID:
    case BPCHAROID:
    case VARCHAROID: {
        if (column_typmod == VARHDRSZ + 1) {
            char val = static_cast<char>(table_data_.get_streamers().value<int8_t>(column_number, row_number));
            return {PointerGetDatum(cstring_to_text_with_len(&val, 1)), false};
        }
    }
    case TEXTOID: {
        auto str = table_data_.get_streamers().value<std::string_view>(column_number, row_number);
        return {PointerGetDatum(cstring_to_text_with_len(str.data(), str.size())), false};
    }
    default: {
        nd::array curr_val = table_data_.get_streamers().get_sample(column_number, row_number);
        if (curr_val.is_none()) {
            return {(Datum)0, true};
        }
        try {
            auto datum = pg::utils::nd_to_datum(std::move(curr_val), base_typeid, column_typmod);
            return {datum, datum == (Datum)0};
        } catch (const pg::exception& e) {
            elog(ERROR,
                 "Fetch Tuple: %s\n Column '%s' has unsupported type: '%s' (base OID %u)",
                 e.message().c_str(),
                 table_data_.get_atttypename(column_number).c_str(),
                 format_type_be(base_typeid),
                 base_typeid);
        }
    }
    }

    return {(Datum)0, true};
}

inline void table_scan::convert_nd_to_pg(int64_t row_number, Datum* values, bool* nulls) const noexcept
{
    for (auto col : null_columns_) {
        nulls[col] = true;
    }
    for (auto col : scored_columns_) {
        nulls[col] = false;
    }
    for (auto col : special_columns_) {
        values[col] = pg::utils::make_special_datum(table_id_, row_number, col, table_data_.get_base_atttypid(col));
        nulls[col] = false;
    }
    for (auto col : process_columns_) {
        auto [datum, is_null] = get_datum(col, row_number);
        values[col] = datum;
        nulls[col] = is_null;
    }
}

inline bool table_scan::get_next_tuple(TupleTableSlot* slot)
{
    if (current_position_ < num_rows_) [[likely]] {
        ExecClearTuple(slot);
        convert_nd_to_pg(current_position_, slot->tts_values, slot->tts_isnull);

        const auto [block_number, offset] = utils::row_number_to_tid(current_position_);
        ItemPointerSet(&slot->tts_tid, block_number, offset);

        slot->tts_tableOid = table_id_;
        // ExecStoreVirtualTuple(slot);
        slot->tts_flags &= ~TTS_FLAG_EMPTY;
        slot->tts_nvalid = slot->tts_tupleDescriptor->natts;

        ++current_position_;
        return true;
    }

    return false;
}

inline void table_scan::fetch_column(AttrNumber column, Datum& value, bool& is_null)
{
    // Check if column should be processed
    auto process_result = impl::should_process_column(table_data_, column, false);
    if (process_result == column_process_result::skip_as_null) {
        is_null = true;
        return;
    } else if (process_result == column_process_result::skip_as_special) {
        is_null = false; // For indexed columns with score
        return;
    }

    auto [datum, is_null_datum] = get_datum(column, current_position_);
    if (is_null_datum) {
        is_null = true;
        return;
    }

    value = datum;
    is_null = false;
}

inline void table_scan::reset_scan()
{
    current_position_ = 0;
}

} // namespace pg
