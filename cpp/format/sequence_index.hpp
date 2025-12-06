#pragma once

#include "format.hpp"
#include "pod_serializable.hpp"
#include "stl_serializable.hpp"

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <cstdint>
#include <vector>

namespace format {

struct sequence_entry
{
    uint32_t length;
    uint32_t offset;
    uint32_t up_to;

    friend bool operator==(const sequence_entry& lhs, const sequence_entry& rhs) noexcept = default;
};

static_assert(std::is_trivial_v<sequence_entry> && std::is_standard_layout_v<sequence_entry>);

struct sequence_index
{
    explicit sequence_index(std::string_view version)
        : version_(version)
    {
    }

    sequence_index(std::string_view version, std::vector<sequence_entry>&& sequence_index)
        : version_(version)
        , sequence_index_(std::move(sequence_index))
    {
    }

    void append(sequence_entry s)
    {
        sequence_index_.push_back(s);
        mark_lengths_dirty();
    }

    void append(uint32_t length)
    {
        if (sequence_index_.empty()) {
            sequence_index_.emplace_back(sequence_entry{length, 0, 0});
            mark_lengths_dirty();
            return;
        }
        if (sequence_index_.back().length == length) {
            sequence_index_.back().up_to += 1;
            mark_lengths_dirty();
            return;
        }
        auto po = sequence_index_.back().offset;
        auto pl = sequence_index_.back().length;
        auto pu = sequence_index_.back().up_to;
        auto ps = sequence_index_.size() == 1 ? pu + 1 : pu - sequence_index_[sequence_index_.size() - 2].up_to;
        auto offset = po + ps * pl;
        sequence_index_.emplace_back(sequence_entry{length, offset, pu + 1});
        mark_lengths_dirty();
    }

    void merge(const sequence_index& other)
    {
        if (other.sequence_index_.empty()) {
            return;
        }

        bool had_cached_lengths = !lengths_dirty_;
        if (had_cached_lengths) {
            const auto& other_lengths = other.get_sequence_lengths();
            sequence_lengths_.reserve(sequence_lengths_.size() + other_lengths.size());
            sequence_lengths_.insert(sequence_lengths_.end(), other_lengths.begin(), other_lengths.end());
        }

        for (const auto& other_entry : other.sequence_index_) {
            uint32_t prev_up_to_other = 0;
            auto it = std::find_if(
                other.sequence_index_.begin(), other.sequence_index_.end(), [&other_entry](const sequence_entry& e) {
                    return &e == &other_entry;
                });
            if (it != other.sequence_index_.begin()) {
                prev_up_to_other = (it - 1)->up_to + 1;
            }

            uint32_t sequences_in_entry = other_entry.up_to - prev_up_to_other + 1;

            for (uint32_t i = 0; i < sequences_in_entry; ++i) {
                append(other_entry.length);
            }
        }

        if (had_cached_lengths) {
            lengths_dirty_ = false;
        }
    }

    const auto& indexes() const noexcept
    {
        return sequence_index_;
    }

    static icm::json to_json(const sequence_index& index)
    {
        icm::json entries;
        for (const auto& entry : index.sequence_index_) {
            entries.push_back({{entry.length, entry.offset, entry.up_to}});
        }
        return entries;
    }

    static sequence_index from_json(const icm::const_json& json)
    {
        std::vector<sequence_entry> entries;
        for (auto item = json.array_begin(); item != json.array_end(); ++item) {
            auto arr = (*item).get<std::array<uint32_t, 3>>();
            entries.push_back(sequence_entry{arr[0], arr[1], arr[2]});
        }
        return sequence_index("", std::move(entries));
    }

    uint32_t size() const noexcept
    {
        if (sequence_index_.empty()) {
            return 0;
        }
        return sequence_index_.back().up_to + 1;
    }

    std::pair<uint32_t, uint32_t> get_sample_range(int sequence_idx) const
    {
        if (sequence_idx < 0 || sequence_index_.empty()) {
            ASSERT_MESSAGE(false, "Invalid sequence index or empty sequence index.");
            return {0, 0};
        }

        const uint32_t idx = static_cast<uint32_t>(sequence_idx);

        size_t left = 0;
        size_t right = sequence_index_.size();

        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (sequence_index_[mid].up_to < idx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if (left >= sequence_index_.size() || sequence_index_[left].up_to < idx) {
            ASSERT_MESSAGE(false, "Invalid sequence index or empty sequence index. Sequence etnry not found.");
            return {0, 0};
        }

        const auto& entry = sequence_index_[left];
        uint32_t prev_up_to = (left > 0) ? sequence_index_[left - 1].up_to + 1 : 0;
        uint32_t sequences_before = idx - prev_up_to;

        uint32_t start_idx = entry.offset + sequences_before * entry.length;
        uint32_t end_idx = start_idx + entry.length;

        return {start_idx, end_idx};
    }

    const std::vector<uint64_t>& get_sequence_lengths() const
    {
        if (lengths_dirty_) {
            update_sequence_lengths();
        }
        return sequence_lengths_;
    }

    void update_sequence(uint32_t sequence_idx, uint32_t new_length)
    {
        if (sequence_index_.empty()) {
            ASSERT_MESSAGE(false, "Cannot update sequence in empty index.");
            return;
        }

        const uint32_t total_sequences = size();
        if (sequence_idx >= total_sequences) {
            ASSERT_MESSAGE(false, "Sequence index out of bounds.");
            return;
        }

        size_t entry_idx = 0;
        while (entry_idx < sequence_index_.size() && sequence_index_[entry_idx].up_to < sequence_idx) {
            entry_idx++;
        }

        if (entry_idx >= sequence_index_.size()) {
            ASSERT_MESSAGE(false, "Failed to find entry for sequence index.");
            return;
        }

        auto& target_entry = sequence_index_[entry_idx];
        if (target_entry.length == new_length) {
            return;
        }

        uint32_t prev_up_to = (entry_idx > 0) ? sequence_index_[entry_idx - 1].up_to + 1 : 0;
        uint32_t position_in_entry = sequence_idx - prev_up_to;
        uint32_t sequences_in_entry = target_entry.up_to - prev_up_to + 1;

        if (sequences_in_entry == 1) {
            target_entry.length = new_length;
            recalculate_offsets_from(entry_idx + 1);
            mark_lengths_dirty();
            return;
        }

        std::vector<sequence_entry> new_entries;
        if (position_in_entry > 0) {
            new_entries.push_back(
                sequence_entry{target_entry.length, target_entry.offset, prev_up_to + position_in_entry - 1});
        }

        uint32_t target_up_to = prev_up_to + position_in_entry;
        uint32_t target_offset = target_entry.offset + position_in_entry * target_entry.length;
        new_entries.push_back(sequence_entry{new_length, target_offset, target_up_to});

        if (position_in_entry + 1 < sequences_in_entry) {
            uint32_t remaining_offset = target_offset + new_length;
            new_entries.push_back(sequence_entry{target_entry.length, remaining_offset, target_entry.up_to});
        }

        sequence_index_.erase(sequence_index_.begin() + entry_idx);
        sequence_index_.insert(sequence_index_.begin() + entry_idx, new_entries.begin(), new_entries.end());

        recalculate_offsets_from(entry_idx + new_entries.size());
        merge_adjacent_entries();
        mark_lengths_dirty();
    }

private:
    void recalculate_offsets_from(size_t start_idx)
    {
        for (size_t i = start_idx; i < sequence_index_.size(); ++i) {
            if (i == 0) {
                sequence_index_[i].offset = 0;
            } else {
                const auto& prev = sequence_index_[i - 1];
                uint32_t prev_start = (i > 1) ? sequence_index_[i - 2].up_to + 1 : 0;
                uint32_t prev_count = prev.up_to - prev_start + 1;
                sequence_index_[i].offset = prev.offset + prev_count * prev.length;
            }
        }
    }

    void merge_adjacent_entries()
    {
        if (sequence_index_.size() <= 1) {
            return;
        }

        std::vector<sequence_entry> merged;
        merged.reserve(sequence_index_.size());

        for (size_t i = 0; i < sequence_index_.size(); ++i) {
            if (merged.empty() || merged.back().length != sequence_index_[i].length) {
                merged.push_back(sequence_index_[i]);
            } else {
                merged.back().up_to = sequence_index_[i].up_to;
            }
        }

        sequence_index_ = std::move(merged);
    }

    void mark_lengths_dirty() const
    {
        lengths_dirty_ = true;
    }

    void update_sequence_lengths() const
    {
        sequence_lengths_.clear();

        if (sequence_index_.empty()) {
            lengths_dirty_ = false;
            return;
        }

        sequence_lengths_.reserve(sequence_index_.back().up_to + 1);

        auto current_sequence = 0u;
        for (const auto& [length, offset, up_to] : sequence_index_) {
            for (; current_sequence <= up_to; ++current_sequence) {
                sequence_lengths_.push_back(static_cast<int64_t>(length));
            }
        }

        lengths_dirty_ = false;
    }

public:
    std::string version_;
    std::vector<sequence_entry> sequence_index_;

private:
    mutable std::vector<uint64_t> sequence_lengths_;
    mutable bool lengths_dirty_ = true;
};

template <>
struct serializable<format::sequence_index>
{
    static read_result<format::sequence_index> read(const base::memory_buffer& bytes, int64_t offset)
    {
        int64_t size = bytes[offset++];
        std::string version(reinterpret_cast<const char*>(&bytes[offset]), size);
        offset += size;
        std::vector<format::sequence_entry> ids;
        while (offset < bytes.size()) {
            auto [id, off] = serializable<format::sequence_entry>::read(bytes, offset);
            ids.emplace_back(id);
            offset = off;
        }
        return {format::sequence_index(std::move(version), std::move(ids)), offset};
    }

    static int64_t output_size(const format::sequence_index& o) noexcept
    {
        return 1 + o.version_.size() + o.sequence_index_.size() * sizeof(format::sequence_entry);
    }

    static void write(const format::sequence_index& o, buffer_t& bytes, int64_t offset)
    {
        bytes[offset++] = static_cast<uint8_t>(o.version_.size());
        for (auto& ch : o.version_) {
            serializable<char>::write(ch, bytes, offset++);
        }
        for (const auto& id : o.sequence_index_) {
            serializable<format::sequence_entry>::write(id, bytes, offset);
            offset += sizeof(format::sequence_entry);
        }
    }
};

} // namespace format
