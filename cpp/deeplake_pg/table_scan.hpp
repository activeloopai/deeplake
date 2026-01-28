#pragma once

#include "table_data.hpp"
#include "utils.hpp"

#include <access/htup.h>
#include <access/parallel.h>
#include <access/tupdesc.h>
#include <postgres.h>
#include <executor/tuptable.h>
#include <miscadmin.h>

#include <vector>

namespace pg {

class table_scan
{
public:
    enum class column_process_result : uint8_t {
        process,
        skip_as_null,
        skip_as_special,
        skip_as_scored
    };

public:
    inline table_scan(Oid table_id, bool is_parallel, bool streamer_only);
    inline table_scan(const table_scan&) = delete;
    inline table_scan(table_scan&&) = default;
    inline table_scan& operator=(const table_scan&) = delete;
    inline table_scan& operator=(table_scan&&) = delete;
    inline ~table_scan() = default;

    inline std::pair<Datum, bool> get_datum(int32_t column_number, int64_t row_number) const noexcept;
    inline void convert_nd_to_pg(int64_t row_number, Datum* values, bool* nulls) const noexcept;

    inline bool get_next_tuple(TupleTableSlot* slot);

    inline void fetch_column(AttrNumber column, Datum& value, bool& is_null);
    inline void reset_scan();

    inline int64_t get_current_position() const noexcept
    {
        return current_position_;
    }

    inline void set_current_position(int64_t position) noexcept
    {
        current_position_ = position;
    }

    inline int64_t get_num_rows() const noexcept
    {
        return num_rows_;
    }

    inline const auto& get_table_data() const noexcept
    {
        return table_data_;
    }

    int32_t nkeys = 0;
    ScanKey keys = 0;

private:
    std::vector<Datum> values_;
    std::vector<uint8_t> nulls_;

    std::vector<int32_t> null_columns_;
    std::vector<int32_t> special_columns_;
    std::vector<int32_t> scored_columns_;
    std::vector<int32_t> process_columns_;
    std::vector<bool> has_streamer_columns_;

    table_data& table_data_;
    int64_t current_position_ = 0;
    int64_t num_rows_ = 0;
    int64_t num_filtered_rows_ = 0;
    Oid table_id_ = InvalidOid;
    bool is_parallel_ = false;
};

} // namespace pg

#include "table_scan_impl.hpp"
