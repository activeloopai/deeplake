#pragma once

#include "exceptions.hpp"
#include "utils.hpp"

#include <base/base.hpp>
#include <deeplake_api/dataset.hpp>
#include <icm/string_map.hpp>
#include <nd/comparison.hpp>
#include <nd/functions.hpp>
#include <nd/norm.hpp>
#include <query_core/order_type.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pg {

enum class command_type
{
    CMD_UNKNOWN,
    CMD_SELECT,
    CMD_UPDATE,
    CMD_INSERT,
    CMD_DELETE,
    CMD_MERGE,
    CMD_UTILITY,
    CMD_NOTHING
};

struct query_info
{
    using limit_t = int32_t;

    static query_info& current()
    {
        static query_info instance;
        return instance;
    }

    static void push_context(QueryDesc* query_desc)
    {
        current_query_desc = query_desc;
    }

    static void pop_context(QueryDesc* query_desc)
    {
        if (current_query_desc != query_desc) {
            elog(WARNING, "query_info::pop_context: Mismatched QueryDesc, expected=%p, got=%p", current_query_desc, query_desc);
            return;
        }

        // Clear the context
        pg::query_info::current().reset();
        current_query_desc = nullptr;
    }

    static void cleanup()
    {
        pg::query_info::pop_context(current_query_desc);
    }

    static bool is_in_executor_context(QueryDesc* query_desc)
    {
        return current_query_desc == query_desc;
    }

    static bool is_in_top_context()
    {
        return current_query_desc == nullptr;
    }

    static auto get_plan_targetlist()
    {
        if (current_query_desc != nullptr) {
            return current_query_desc->plannedstmt->planTree->targetlist;
        }
        return NIL;
    }

public:
    inline void reset()
    {
        is_count_star_ = false;
        has_deferred_fetch_ = false;
        is_deeplake_table_referenced_ = false;
        are_all_tables_deeplake_ = false;
        limit_ = std::numeric_limits<limit_t>::max();
        cmd_type_ = command_type::CMD_UNKNOWN;
        reset_current_score();
    }

    inline void set_limit(limit_t limit) noexcept
    {
        limit_ = limit;
    }

    inline limit_t limit() const noexcept
    {
        return limit_;
    }

    inline void set_command_type(command_type cmd_type) noexcept
    {
        cmd_type_ = cmd_type;
    }

    inline enum command_type command_type() const noexcept
    {
        return cmd_type_;
    }

    inline void set_count_star(bool is_count_star) noexcept
    {
        is_count_star_ = is_count_star;
    }

    inline bool is_count_star() const noexcept
    {
        return is_count_star_;
    }

    inline void set_has_deferred_fetch(bool has_deferred_fetch) noexcept
    {
        has_deferred_fetch_ = has_deferred_fetch;
    }

    inline bool has_deferred_fetch() const noexcept
    {
        return has_deferred_fetch_;
    }

    inline bool receiver_registered() const noexcept
    {
        return has_deferred_fetch();
    }

    bool has_current_score() const noexcept
    {
        return current_score_ != std::numeric_limits<float>::max();
    }

    float current_score() const noexcept
    {
        return current_score_;
    }

    void set_current_score(float score) noexcept
    {
        current_score_ = score;
    }

    void reset_current_score() noexcept
    {
        current_score_ = std::numeric_limits<float>::max();
    }

    inline void set_is_deeplake_table_referenced(bool is_referenced) noexcept
    {
        is_deeplake_table_referenced_ = is_referenced;
    }

    inline bool is_deeplake_table_referenced() const noexcept
    {
        return is_deeplake_table_referenced_;
    }

    inline void set_all_tables_are_deeplake(bool all_deeplake) noexcept
    {
        are_all_tables_deeplake_ = all_deeplake;
    }

    inline bool are_all_tables_deeplake() const noexcept
    {
        return are_all_tables_deeplake_;
    }

private:
    static QueryDesc* current_query_desc;

    bool is_count_star_ = false;
    bool has_deferred_fetch_ = false;
    bool is_deeplake_table_referenced_ = false;
    bool are_all_tables_deeplake_ = false;
    limit_t limit_ = std::numeric_limits<limit_t>::max();
    enum command_type cmd_type_ = command_type::CMD_UNKNOWN;
    float current_score_ = std::numeric_limits<float>::max();
};

struct index_options
{
    std::string index_type;
};

/// Wrapper around dataset
struct index_info
{
    static index_info& current()
    {
        static index_info instance;
        return instance;
    }

    void reset()
    {
        table_name_.clear();
        column_names_.clear();
        index_name_.clear();
        index_type_ = deeplake_core::deeplake_index_type::type::none;
        order_type_ = query_core::order_type::descending;
    }

    void create();

    const std::shared_ptr<deeplake_api::dataset>& dataset() const;

    inline void set_table_name(std::string_view table_name) noexcept
    {
        table_name_ = table_name;
    }

    inline const std::string& table_name() const noexcept
    {
        return table_name_;
    }

    inline void set_column_name(std::string_view column_name) noexcept
    {
        auto pos = std::string_view::npos;
        do {
            pos = column_name.find(',');
            if (pos != std::string_view::npos) {
                if (pos != 0) {
                    column_names_.emplace_back(column_name.substr(0, pos));
                }
                column_name = column_name.substr(pos + 1);
            }
        } while (pos != std::string_view::npos);
        if (!column_name.empty()) {
            column_names_.emplace_back(column_name);
        }
    }

    inline void set_column_names(std::vector<std::string>&& names) noexcept
    {
        column_names_ = std::move(names);
    }

    inline const std::string& column_name() const
    {
        return column_names_.at(0);
    }

    inline const std::vector<std::string>& column_names() const noexcept
    {
        return column_names_;
    }

    inline std::string get_column_names_string() const noexcept
    {
        std::string column_names_string;
        for (const auto& column_name : column_names_) {
            column_names_string += column_name + ",";
        }
        if (!column_names_string.empty()) {
            column_names_string.pop_back();
        }
        return column_names_string;
    }

    inline void set_index_name(std::string_view index_name) noexcept
    {
        index_name_ = index_name;
    }

    inline const std::string& index_name() const noexcept
    {
        return index_name_;
    }

    inline void set_index_type(std::string_view index_type)
    {
        index_type_ = deeplake_core::deeplake_index_type::from_string(index_type);
        if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
            elog(WARNING, "Unsupported index type: %s", index_type.data());
        }
    }

    inline deeplake_core::deeplake_index_type::type index_type() const noexcept
    {
        return index_type_;
    }

    inline void set_order_type(query_core::order_type o) noexcept
    {
        order_type_ = o;
    }

    inline auto order_type() const noexcept
    {
        return order_type_;
    }

    inline bool is_hybrid_index() const noexcept
    {
        return column_names_.size() > 1;
    }

    inline bool is_numeric_index() const noexcept
    {
        if (column_names_.size() != 1) {
            return false;
        }
        const auto& col = dataset()->get_column(column_name());
        return !col.type().data_type().is_array() && nd::dtype_is_numeric(col.type().data_type().get_dtype());
    }

    bool can_run_query(const query_core::top_k_search_info& info, const std::string& column_name) const
    {
        auto& col = dataset()->get_column(column_name);
        auto index_holder = col.index_holder();
        if (index_holder == nullptr) {
            return false;
        }
        return index_holder->can_run_query(info);
    }

    query_core::query_result run_query(query_core::top_k_search_info info, const std::string& column_name)
    {
        auto& col = dataset()->get_column(column_name);
        auto index_holder = col.index_holder();
        if (index_holder == nullptr) {
            return {};
        }
        return async::run_on_main([i = std::move(info), idx = std::move(index_holder)]() mutable {
            return idx->run_query(i, {}, {}).then([](auto&& res) {
                return std::move(res).front();
            });
        }).get_future().get();
    }

    bool can_run_query(const query_core::inverted_index_search_info& info) const
    {
        auto& col = dataset()->get_column(column_name());
        auto index_holder = col.index_holder();
        if (index_holder == nullptr) {
            return false;
        }
        return index_holder->can_run_query(info);
    }

    icm::roaring run_query(const query_core::inverted_index_search_info& info)
    {
        auto& col = dataset()->get_column(column_name());
        auto index_holder = col.index_holder();
        ASSERT(index_holder != nullptr);
        auto indices = index_holder->run_query(info).get_future().get();
        ASSERT(indices.size() == 1);
        return indices.front();
    }

    bool can_run_query(const query_core::text_search_info& info) const
    {
        auto& col = dataset()->get_column(column_name());
        auto index_holder = col.index_holder();
        if (index_holder == nullptr) {
            return false;
        }
        return index_holder->can_run_query(info);
    }

    icm::roaring run_query(const query_core::text_search_info& info)
    {
        auto& col = dataset()->get_column(column_name());
        auto index_holder = col.index_holder();
        ASSERT(index_holder != nullptr);
        auto indices = index_holder->run_query(info).get_future().get();
        ASSERT(indices.size() == 1);
        return indices.front();
    }

    void collect_result(query_core::query_result result, std::vector<std::pair<ItemPointerData, float>>& res, bool sort) const noexcept
    {
        res.reserve(result.indices.size());
        {
            const bool has_score = (result.scores && result.scores.size() == result.indices.size());
            for (auto i = 0; i < result.indices.size(); ++i) {
                auto rid = result.indices[i];
                auto score = has_score ? result.scores[i].value<float>(0) : 0.0f;
                auto [block_id, offset] = utils::row_number_to_tid(rid);
                ItemPointerData item_pointer;
                ItemPointerSet(&item_pointer, block_id, offset);
                res.emplace_back(item_pointer, score);
            }
        }
        if (sort) {
            if (order_type_ == query_core::order_type::descending) {
                std::sort(res.begin(), res.end(), [&](auto a, auto b) {
                    return a.second > b.second;
                });
            } else {
                std::sort(res.begin(), res.end(), [&](auto a, auto b) {
                    return a.second < b.second;
                });
            }
        }
    }

    void create_deeplake_indexes();
    void drop_deeplake_indexes();

private:
    std::string table_name_;
    std::vector<std::string> column_names_;
    std::string index_name_;
    deeplake_core::deeplake_index_type::type index_type_ = deeplake_core::deeplake_index_type::type::none;
    query_core::order_type order_type_ = query_core::order_type::descending;
};

class pg_index
{
public:
    /// @name static member functions
    /// @{

    /// singleton instance
    static pg_index& instance()
    {
        static pg_index instance;
        return instance;
    }

    static index_info& get_index_info(Oid oid)
    {
        auto it = instance().indexes_.find(oid);
        if (it == instance().indexes_.end()) {
            throw pg::exception("Index not found");
        }
        return it->second;
    }

    static void create_index_info(Oid oid)
    {
        try {
            index_info::current().create();
        } catch (const std::exception& e) {
            const auto idx_name = index_info::current().index_name();
            /// TODO: Swich to error after fixing the issue in Deeplake
            elog(WARNING, "Error while creating index '%s':\n%s", idx_name.c_str(), e.what());
        }
        instance().indexes_.emplace(oid, index_info::current());
        index_info::current().reset();
    }

    static bool has_index_info(Oid oid)
    {
        return instance().indexes_.find(oid) != instance().indexes_.end();
    }

    static auto& get_indexes()
    {
        return instance().indexes_;
    }

    static bool has_indexes()
    {
        return !instance().indexes_.empty();
    }

    static bool has_index_created_on_table(const std::string& table_name)
    {
        for (const auto& [oid, idx_info] : get_indexes()) {
            if (idx_info.table_name() == table_name) {
                return true;
            }
        }
        return false;
    }

    static Oid get_oid(const std::string& table_name, const std::string& column_name)
    {
        for (const auto& [oid, idx_info] : get_indexes()) {
            if (!idx_info.is_hybrid_index() && idx_info.table_name() == table_name && idx_info.column_name() == column_name) {
                return oid;
            }
        }
        return InvalidOid;
    }

    static void erase_info(const std::string& index_name)
    {
        for (auto it = instance().indexes_.begin(); it != instance().indexes_.end();) {
            if (it->second.index_name() == index_name) {
                it->second.drop_deeplake_indexes();
                it = instance().indexes_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static void erase_table_info(const std::string& table_name)
    {
        for (auto it = instance().indexes_.begin(); it != instance().indexes_.end();) {
            if (it->second.table_name() == table_name) {
                it = instance().indexes_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static void erase_column_info(const std::string& table_name, const std::string& column_name)
    {
        for (auto it = instance().indexes_.begin(); it != instance().indexes_.end();) {
            if (it->second.table_name() == table_name && it->second.column_name() == column_name) {
                it = instance().indexes_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static void clear()
    {
        instance().indexes_.clear();
    }

    /// @}

    float deeplake_cosine_similarity(const pg::array_type& f, const pg::array_type& s)
    {
        try {
            return nd::cosine_similarity(f, s).value<float>(0);
        } catch (const base::exception& e) {
            base::log_warning(base::log_channel::index, "Cosine similarity failed: {}", e.what());
        }
        return .0f;
    }

    bool vector_lt(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l < r);
    }

    bool vector_le(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l <= r);
    }

    bool vector_eq(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l == r);
    }

    bool vector_ne(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l != r);
    }

    bool vector_ge(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l >= r);
    }

    bool vector_gt(const pg::array_type& l, const pg::array_type& r)
    {
        return nd::all(l > r);
    }

    int64_t vector_compare(const pg::array_type& l, const pg::array_type& r)
    {
        if (vector_eq(l, r)) {
            return 0;
        }
        if (vector_lt(l, r)) {
            return -1;
        }
        return 1;
    }

    float maxsim(const pg::array_type& f, const pg::array_type& s)
    {
        try {
            return nd::maxsim(f, s).value<float>(0);
        } catch (const base::exception& e) {
            base::log_warning(base::log_channel::index, "MaxSim failed: {}", e.what());
        }
        return .0f;
    }

private:
    pg_index() = default;

    std::unordered_map<Oid, index_info> indexes_;
};

void erase_indexer_data(const std::string& table_name,
                        const std::string& column_name,
                        const std::string& index_name);
void save_index_metadata(Oid oid);
void load_index_metadata();
void init_deeplake();

} // namespace pg
