#pragma once

#include "table_data.hpp"

namespace pg {

// Session-level credentials management
struct session_credentials
{
    static char* creds_guc_string;     // GUC string variable for credentials
    static char* root_path_guc_string; // GUC string variable for root path

    // Get credentials from current session
    static icm::string_map<> get_credentials();

    // Get root path from current session (returns empty string if not set)
    static std::string get_root_path();

    // Initialize GUC parameters
    static void initialize_guc();
};

struct table_options
{
    static table_options& current()
    {
        static table_options instance;
        return instance;
    }

    void reset()
    {
        dataset_path_.clear();
    }

    inline const std::string& dataset_path() const noexcept
    {
        return dataset_path_;
    }

    inline void set_dataset_path(const std::string& dataset_path) noexcept
    {
        dataset_path_ = dataset_path;
    }

private:
    std::string dataset_path_;
};

class table_storage
{
public:
    static table_storage& instance()
    {
        static table_storage storage;
        if (!storage.tables_loaded_) {
            storage.load_table_metadata();
        }
        return storage;
    }

    void create_table(const std::string& table_name, Oid table_id, TupleDesc tupdesc);
    void drop_table(const std::string& table_name);

    inline bool table_exists(Oid table_id) const noexcept
    {
        return tables_.contains(table_id);
    }

    inline table_data* get_table_data_if_exists(Oid table_id) noexcept
    {
        auto it = tables_.find(table_id);
        if (it != tables_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    inline table_data* get_table_data_if_exists(const std::string& table_name) noexcept
    {
        for (auto& [_, table] : tables_) {
            if (table.get_table_name() == table_name) {
                return &table;
            }
        }
        return nullptr;
    }

    inline void reset_requested_columns() noexcept
    {
        for (auto& [_, table] : tables_) {
            table.reset_requested_columns();
        }
    }

    inline bool table_exists(const std::string& table_name) const noexcept
    {
        for (const auto& [_, table] : tables_) {
            if (table.get_table_name() == table_name) {
                return true;
            }
        }
        return false;
    }

    inline bool erase_table(const std::string& table_name)
    {
        for (auto it = tables_.begin(); it != tables_.end(); ++it) {
            if (it->second.get_table_name() == table_name) {
                tables_.erase(it);
                up_to_date_ = false;
                return true;
            }
        }
        return false;
    }

    inline bool empty() const noexcept
    {
        return tables_.empty();
    }

    inline void refresh()
    {
        for (auto& [_, table_data] : tables_) {
            table_data.refresh();
        }
    }

    inline void refresh_table(Oid table_id)
    {
        auto it = tables_.find(table_id);
        if (it != tables_.end()) {
            it->second.refresh();
        }
    }

    // Data operations
    void insert_slot(Oid table_id, TupleTableSlot* slot);
    void insert_slots(Oid table_id, int32_t nslots, TupleTableSlot** slots);
    bool delete_tuple(Oid table_id, ItemPointer tid);
    bool update_tuple(Oid table_id, ItemPointer tid, HeapTuple new_tuple);
    bool fetch_tuple(Oid table_id, ItemPointer tid, TupleTableSlot* slot);
    void mark_subxact_change(Oid table_id);
    void rollback_subxact(SubTransactionId sub_id);
    void commit_subxact(SubTransactionId sub_id, SubTransactionId parent_sub_id);

    bool flush_all()
    {
        /// Flush all inserts
        for (auto& [_, table_data] : tables_) {
            if (!table_data.flush()) {
                return false;
            }
        }
        return true;
    }

    void commit_all()
    {
        /// Flush all inserts
        for (auto& [_, table_data] : tables_) {
            table_data.commit();
        }
        subxact_snapshots_.clear();
    }

    void rollback_all()
    {
        // Rollback all changes in all tables
        for (auto& [_, table_data] : tables_) {
            table_data.rollback();
        }
        subxact_snapshots_.clear();
    }

    inline auto& get_tables() noexcept
    {
        return tables_;
    }

    // Cleanup
    void clear();

    table_data& get_table_data(Oid table_id);
    table_data& get_table_data(const std::string& table_name);

    /// Called from indexer loading
    void load_table_metadata();
    /// Re-read metadata to discover newly created tables.
    /// Keeps existing table_data objects (with open dataset connections) intact.
    void force_load_table_metadata()
    {
        tables_loaded_ = false;
        load_table_metadata();
    }

    /// Clear all state and reload from scratch. Use when the root_path changes
    /// and all existing table_data entries are stale.
    void reset_and_load_table_metadata()
    {
        tables_.clear();
        views_.clear();
        tables_loaded_ = false;
        ddl_log_last_seq_ = 0;
        load_table_metadata();
    }
    void mark_metadata_stale() noexcept
    {
        tables_loaded_ = false;
        up_to_date_ = false;
    }
    void load_views();

    inline const auto& get_views() const noexcept
    {
        return views_;
    }

    inline bool view_exists(Oid view_oid) const noexcept
    {
        return views_.contains(view_oid);
    }

    void add_view(Oid view_oid, const std::string& view_name, const std::string& view_str);
    void erase_view(const std::string& view_name);

    inline const std::string& get_schema_name() const noexcept
    {
        return schema_name_;
    }

    inline void set_schema_name(std::string&& schema_name) noexcept
    {
        up_to_date_ = false;
        schema_name_ = std::move(schema_name);
    }

    void load_schema_name();

    inline bool is_up_to_date() const noexcept
    {
        return up_to_date_;
    }

    inline void set_up_to_date(bool up_to_date) noexcept
    {
        up_to_date_ = up_to_date;
    }

    /**
     * RAII guard to suppress auto-creation of tables during load_table_metadata().
     *
     * When creating a table concurrently, we don't want load_table_metadata() to
     * auto-create tables from the catalog, as this causes race conditions with
     * other backends that are also creating tables.
     *
     * Usage: Create this guard before calling table_storage::instance() during DDL.
     */
    class ddl_context_guard
    {
    public:
        ddl_context_guard()
        {
            in_ddl_context_ = true;
        }
        ~ddl_context_guard()
        {
            in_ddl_context_ = false;
        }
        ddl_context_guard(const ddl_context_guard&) = delete;
        ddl_context_guard& operator=(const ddl_context_guard&) = delete;
    };

    static bool in_ddl_context() noexcept
    {
        return in_ddl_context_;
    }

    /**
     * RAII guard for catalog-only table creation (no S3 dataset operations).
     *
     * When syncing tables from the S3 catalog, the datasets already exist remotely.
     * We only need the pg_class entry so the PG parser can find them.
     * This guard makes create_table() skip exists-check, open_dataset, column
     * creation, and commit — reducing CREATE TABLE from ~6.5s to ~0ms.
     */
    class catalog_only_guard
    {
    public:
        catalog_only_guard()
        {
            catalog_only_create_ = true;
        }
        ~catalog_only_guard()
        {
            catalog_only_create_ = false;
        }
        catalog_only_guard(const catalog_only_guard&) = delete;
        catalog_only_guard& operator=(const catalog_only_guard&) = delete;
    };

    static bool is_catalog_only_create() noexcept
    {
        return catalog_only_create_;
    }

    static void set_catalog_only_create(bool value) noexcept
    {
        catalog_only_create_ = value;
    }

private:
    static inline thread_local bool in_ddl_context_ = false;
    static inline thread_local bool catalog_only_create_ = false;
    table_storage() = default;

    void save_table_metadata(const table_data& td);
    void erase_table_metadata(const std::string& table_name);

    std::unordered_map<Oid, table_data> tables_;
    std::unordered_map<SubTransactionId, std::unordered_map<Oid, table_data::tx_snapshot>> subxact_snapshots_;
    std::unordered_map<Oid, std::pair<std::string, std::string>> views_;
    std::string schema_name_ = "public";
    bool tables_loaded_ = false;
    bool up_to_date_ = true;
    int64_t ddl_log_last_seq_ = 0;
};

} // namespace pg
