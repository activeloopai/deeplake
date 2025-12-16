#pragma once

#include <query_core/search_config.hpp>

namespace tql {

struct options
{
    bool allow_nested_query = false;
    bool allow_selection_list = false;
    bool allow_group = false;
    bool allow_ungroup = false;
    bool use_index = false;
    bool parsing_only = false;

    // Search configuration for vector similarity searches
    query_core::search_config search_config = query_core::search_config::default_config();

    static options default_config();

    static options one_level_filter_only();

    static options filter_only();

    static options linear_only();

    static options allow_everything();

    static options web_environment();
};

}
