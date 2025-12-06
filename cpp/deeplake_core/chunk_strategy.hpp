#pragma once

#include <base/assert.hpp>
#include <base/random.hpp>
#include <nd/array.hpp>
#include <nd/type.hpp>

#include <cstdint>
#include <ranges>
#include <string>
#include <variant>

namespace icm {
class json;
class const_json;
} // namespace icm

namespace deeplake_core {

struct num_rows_chunk_strategy
{
    num_rows_chunk_strategy() = default;

    num_rows_chunk_strategy(int64_t num_rows);

    static num_rows_chunk_strategy from_json(const icm::const_json& json);

    icm::json to_json() const;

    [[nodiscard]] int64_t num_rows() const
    {
        return num_rows_;
    }

    std::vector<int64_t> chunk_ranges(const std::ranges::range auto& rows) const
    {
        ASSERT(num_rows_ > 0);
        std::vector<int64_t> ranges;
        ranges.reserve(rows.size() / num_rows_ + 2);
        for (int64_t i = 0; i < rows.size(); i += num_rows_) {
            ranges.push_back(i);
        }
        ranges.push_back(rows.size());
        return ranges;
    }

    bool operator==(const num_rows_chunk_strategy& other) const noexcept = default;

private:
    int64_t num_rows_ = 0L;
};

struct num_bytes_chunk_strategy
{
    num_bytes_chunk_strategy() = default;

    num_bytes_chunk_strategy(int64_t num_bytes);

    static num_bytes_chunk_strategy from_json(const icm::const_json& json);

    icm::json to_json() const;

    [[nodiscard]] int64_t num_bytes() const
    {
        return num_bytes_;
    }

    std::vector<int64_t> chunk_ranges(const std::ranges::range auto& rows) const
    {
        if (rows.empty()) {
            return {};
        }
        const auto v = get_single_row_estimated_bytes(rows);
        ASSERT(v > 0);
        const auto num_rows = std::max(static_cast<int64_t>(num_bytes_ / v), static_cast<int64_t>(1));

        num_rows_chunk_strategy t(num_rows);
        return t.chunk_ranges(rows);
    }

    bool operator==(const num_bytes_chunk_strategy& other) const noexcept = default;

private:
    int64_t get_single_row_estimated_bytes(const std::ranges::range auto& rows) const
    {
        ASSERT(!rows.empty());
        auto sample_count = std::min<std::size_t>(100, rows.size());
        auto& random_number_generator = base::random_engine();
        std::uniform_int_distribution<std::size_t> indices(0, rows.size() - 1);

        int64_t sum = 0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < sample_count; ++i) {
            const auto index = indices(random_number_generator);
            ASSERT(index < rows.size());
            const auto volume = rows[index].volume();
            if (volume != 0) {
                sum += nd::dtype_bytes(rows[index].dtype()) * volume;
                ++count;
            }
        }
        if (count > 0) {
            return sum / count;
        }
        return nd::dtype_bytes(rows[0].dtype());
    }

private:
    int64_t num_bytes_ = 8'000'000L;
};

enum class chunk_strategy_type
{
    num_bytes,
    num_rows
};

class chunk_strategy
{
public:
    chunk_strategy() = default;

    static chunk_strategy num_rows(int64_t num_rows)
    {
        return chunk_strategy(num_rows_chunk_strategy(num_rows));
    }

    static chunk_strategy num_bytes(int64_t num_bytes)
    {
        return chunk_strategy(num_bytes_chunk_strategy(num_bytes));
    }

    static chunk_strategy default_strategy_for_type(const nd::type& type)
    {
        return num_bytes(8'000'000);
    }

    static chunk_strategy default_strategy()
    {
        return num_bytes(8'000'000);
    }

    static chunk_strategy from_json(const icm::const_json& json);

    icm::json to_json() const;

public:
    chunk_strategy_type get_type() const
    {
        return static_cast<chunk_strategy_type>(data_.index());
    }

    const num_rows_chunk_strategy& as_num_rows_strategy() const
    {
        return std::get<num_rows_chunk_strategy>(data_);
    }

    const num_bytes_chunk_strategy& as_num_bytes_strategy() const
    {
        return std::get<num_bytes_chunk_strategy>(data_);
    }

public:
    std::vector<int64_t> chunk_ranges(const std::ranges::range auto& rows) const
    {
        return std::visit(
            [&rows](const auto& strategy) {
                return strategy.chunk_ranges(rows);
            },
            data_);
    }

    bool operator==(const chunk_strategy& other) const noexcept = default;

private:
    chunk_strategy(num_bytes_chunk_strategy strategy)
        : data_(std::move(strategy))
    {
    }

    chunk_strategy(num_rows_chunk_strategy strategy)
        : data_(std::move(strategy))
    {
    }

    using data_type = std::variant<num_bytes_chunk_strategy, num_rows_chunk_strategy>;
    data_type data_;
};

} // namespace deeplake_core
