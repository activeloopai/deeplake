#pragma once

#include "query_result.hpp"
#include "static_data_t.hpp"
#include "text_search_info.hpp"
#include "top_k_search_info.hpp"
#include "inverted_index_search_info.hpp"

#include <async/promise.hpp>
#include <deeplake_core/index_type.hpp>
#include <icm/roaring.hpp>

#include <memory>
#include <vector>

namespace query_core {

class index_holder
{
public:
    virtual ~index_holder() = default;

    virtual void reset_index_data() = 0;

    virtual std::vector<deeplake_core::index_type> get_indexes() const = 0;

    virtual std::string to_string() const = 0;

public:
    virtual bool can_run_query(const top_k_search_info& info) const = 0;
    virtual bool can_run_query(const text_search_info& info) const = 0;

    virtual bool can_run_query(const inverted_index_search_info& info) const = 0;

    virtual async::promise<query_results>
    run_query(const top_k_search_info& info, const static_data_t& data, std::shared_ptr<const icm::roaring> filter) = 0;

    virtual async::promise<std::vector<icm::roaring>> run_query(const text_search_info& info) = 0;

    virtual async::promise<std::vector<icm::roaring>> run_query(const inverted_index_search_info& info) = 0;
    /// @}
};

} // namespace query_core
