#pragma once

#include "sample.hpp"

#include <nd/stride.hpp>

#include <cstdint>
#include <vector>

namespace deeplake_core {

class sample_range
{
public:
    sample_range(const batch& batch, int64_t begin, int64_t end)
        : batch_(&batch), indices_(icm::index_mapping_t<int64_t>::slice({begin, end, 1L}))
    {
    }

    explicit sample_range(const batch& batch)
        : batch_(&batch), indices_(icm::index_mapping_t<int64_t>::trivial(batch.size()))
    {
    }

public:
    sample operator[](int64_t index) const
    {
        return sample(*batch_, indices_[index]);
    }

    using iterator =
        icm::index_based_iterator<sample_range, sample, icm::use_container_index_tag, int64_t>;

    inline iterator begin() const
    {
        return iterator(*this, 0L);
    }

    inline iterator end() const
    {
        return iterator(*this, indices_.size());
    }

    inline iterator cbegin() const
    {
        return begin();
    }

    inline iterator cend() const
    {
        return end();
    }

    inline const batch& get_batch() const
    {
        ASSERT(batch_ != nullptr);
        return *batch_;
    }

    inline nd::array column(int column_index) const
    {
        auto a = batch_->columns()[column_index].array();
        return nd::stride(std::move(a), indices_);
    }

    inline sample_range filter(std::vector<int64_t>&& indices) const
    {
        auto r = *this;
        r.indices_ =
            icm::index_mapping_t<int64_t>::chain(r.indices_, icm::index_mapping_t<int64_t>::list(std::move(indices)));
        return r;
    }

    inline int64_t size() const
    {
        return indices_.size();
    }

    inline const auto& indices() const
    {
        return indices_;
    }

private:
    const batch* batch_;
    icm::index_mapping_t<int64_t> indices_;
};

} // namespace deeplake_core
