#include "pg_deeplake.hpp"
#include "dl_wal.hpp"
#include "logger.hpp"
#include "table_storage.hpp"
#include "utils.hpp"

#include <deeplake_api/deeplake_api.hpp>
#include <deeplake_core/deeplake_index_type.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <access/xact.h>
#include <commands/dbcommands.h>
#include <miscadmin.h>
#include <storage/ipc.h>

#ifdef __cplusplus
}
#endif

namespace {

// Exit handler that uses _exit() to avoid C++ static destructor crashes.
// PostgreSQL background workers (autovacuum, parallel workers, etc.) can crash
// during normal exit when C++ static objects are destroyed in unpredictable order.
void deeplake_quick_exit(int code, Datum arg)
{
    _exit(code);
}

} // anonymous namespace

namespace pg {

QueryDesc* query_info::current_query_desc = nullptr;

void index_info::create_deeplake_indexes()
{
    auto ds = dataset();
    ASSERT(ds != nullptr);
    bool index_created = false;
    for (auto& column_name : column_names_) {
        /// 'none' means hybrid index type
        if (is_hybrid_index()) {
            index_type_ = deeplake_core::deeplake_index_type::type::none;
        } else if (pg_index::get_oid(table_name_, column_name) != InvalidOid) {
            ereport(
                ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("Failed to create index on table '%s', column '%s'", table_name_.c_str(), column_name.c_str()),
                 errdetail("Multiple indexes on the same column are not supported yet."),
                 errhint("Drop the existing index before creating a new one.")));
        }
        auto& column = ds->get_column(column_name);
        auto index_holder = column.index_holder();
        if (index_holder != nullptr) {
            auto existing_idx_type = deeplake_core::deeplake_index_type::from_string(index_holder->to_string());
            if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                index_type_ = existing_idx_type;
            }
            if (is_hybrid_index() && column.type().data_type().get_dtype() == nd::dtype::string &&
                existing_idx_type != deeplake_core::deeplake_index_type::type::bm25) {
                elog(ERROR,
                     "Conflicting index types for column '%s' in table '%s': existing index type '%s', new index type "
                     "'bm25'",
                     column_name.c_str(),
                     table_name_.c_str(),
                     deeplake_core::deeplake_index_type::to_string(existing_idx_type).data());
            }
            if (index_type_ != existing_idx_type) {
                elog(ERROR,
                     "Conflicting index types for column '%s' in table '%s': existing index type '%s', new index type "
                     "'%s'",
                     column_name.c_str(),
                     table_name_.c_str(),
                     deeplake_core::deeplake_index_type::to_string(index_type_).data(),
                     deeplake_core::deeplake_index_type::to_string(existing_idx_type).data());
            }
            continue;
        }
        auto column_type_kind = column.type().kind();
        if (column_type_kind == deeplake_core::type_kind::text) {
            // Determine text index type
            if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                index_type_ = deeplake_core::deeplake_index_type::type::bm25;
            }
            if (index_type_ != deeplake_core::deeplake_index_type::type::bm25 &&
                index_type_ != deeplake_core::deeplake_index_type::type::exact_text &&
                index_type_ != deeplake_core::deeplake_index_type::type::inverted_index) {
                elog(ERROR,
                     "Invalid index type '%s' for text column. Use 'inverted', 'bm25' or 'exact_text'",
                     deeplake_core::deeplake_index_type::to_string(index_type_).data());
            }
            // Set indexing mode to always for BM25 indexes only
            if (index_type_ == deeplake_core::deeplake_index_type::type::bm25) {
                ds->set_indexing_mode(deeplake::indexing_mode::always);
            }
            column.create_index(deeplake_core::index_type(deeplake_core::text_index_type(index_type_)));
            index_created = true;
        } else if (column_type_kind == deeplake_core::type_kind::generic && !column.type().data_type().is_array() &&
                   nd::dtype_is_numeric(column.type().data_type().get_dtype())) {
            if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                index_type_ = deeplake_core::deeplake_index_type::type::inverted_index;
            }
            column.create_index(deeplake_core::index_type(deeplake_core::numeric_index_type(index_type_)));
            index_created = true;
        } else if (column_type_kind == deeplake_core::type_kind::embedding ||
                   column_type_kind == deeplake_core::type_kind::generic) {
            if (!column.type().data_type().is_array() ||
                column.type().data_type().get_scalar_type().get_dtype() != nd::dtype::float32) {
                elog(
                    ERROR, "Column %s is not float32 array, hence not supported yet for indexing", column_name.c_str());
            }
            const auto dimensions = column.type().data_type().dimensions();
            if (dimensions == 1) {
                // Determine embedding index type
                if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                    index_type_ = deeplake_core::deeplake_index_type::type::clustered;
                }
                if (index_type_ != deeplake_core::deeplake_index_type::type::clustered &&
                    index_type_ != deeplake_core::deeplake_index_type::type::clustered_quantized &&
                    index_type_ != deeplake_core::deeplake_index_type::type::pooled_quantized) {
                    elog(ERROR,
                         "Invalid index type '%s' for embedding column. Use 'clustered', 'clustered_quantized', or "
                         "'pooled_quantized'",
                         deeplake_core::deeplake_index_type::to_string(index_type_).data());
                }
                column.create_index(deeplake_core::index_type(deeplake_core::embedding_index_type(index_type_)));
            } else if (dimensions == 2) {
                if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                    index_type_ = deeplake_core::deeplake_index_type::type::pooled_quantized;
                }
                column.create_index(deeplake_core::index_type(deeplake_core::embeddings_matrix_index_type()));
            } else {
                elog(ERROR,
                     "Column %s has dimensions %d, which is not supported yet for indexing",
                     column_name.c_str(),
                     dimensions);
            }
            index_created = true;
        } else if (column_type_kind == deeplake_core::type_kind::dict) {
            if (index_type_ == deeplake_core::deeplake_index_type::type::none) {
                index_type_ = deeplake_core::deeplake_index_type::type::inverted_index;
            }
            column.create_index(deeplake_core::index_type(deeplake_core::json_index_type(index_type_)));
            index_created = true;
        } else {
            elog(ERROR, "Column %s is not supported yet for indexing", column_name.c_str());
        }
    }
    if (is_hybrid_index()) {
        index_type_ = deeplake_core::deeplake_index_type::type::none;
    }
    if (index_created) {
        table_storage::instance().get_table_data(table_name_).commit(true);
    }
}

void index_info::drop_deeplake_indexes()
{
    auto& ds = dataset();
    ASSERT(ds != nullptr);
    bool index_dropped = false;
    for (auto& column_name : column_names_) {
        auto& column = ds->get_column(column_name);
        auto index_holder = column.index_holder();
        if (index_holder == nullptr) {
            continue;
        }
        auto deeplake_indexes = index_holder->get_indexes();
        for (const auto& index : deeplake_indexes) {
            elog(DEBUG1,
                 "Dropping index on column '%s' in table '%s', type '%s'",
                 column_name.c_str(),
                 table_name_.c_str(),
                 index.to_string().data());
            async::run_on_main([&column, index]() {
                column.drop_index(index);
            })
                .get_future()
                .get();
            index_dropped = true;
        }
    }
    if (index_dropped) {
        table_storage::instance().get_table_data(table_name_).commit(false);
    }
}

const std::shared_ptr<deeplake_api::dataset>& index_info::dataset() const
{
    auto& deeplake_table_data = table_storage::instance().get_table_data(table_name_);
    return deeplake_table_data.get_dataset();
}

void index_info::create()
{
    base::log_debug(base::log_channel::index, "Create index info");
    ASSERT(!table_name_.empty());
    ASSERT(!column_names_.empty());
    ASSERT(table_storage::instance().table_exists(table_name_));
    create_deeplake_indexes();
}

/// @name Index metadata management
/// @description: Functions to manage index metadata in the database
/// @{

void erase_indexer_data(const std::string& table_name, const std::string& column_name, const std::string& index_name)
{
    if (!pg::utils::check_table_exists("pg_deeplake_metadata")) {
        return;
    }
    pg::utils::memory_context_switcher context_switcher;
    StringInfoData buf;
    initStringInfo(&buf);
    if (!index_name.empty()) {
        appendStringInfo(&buf,
                         "DELETE FROM public.pg_deeplake_metadata WHERE index_name = %s",
                         quote_literal_cstr(index_name.c_str()));
    } else if (!table_name.empty() && !column_name.empty()) {
        appendStringInfo(&buf,
                         "DELETE FROM public.pg_deeplake_metadata WHERE table_name = %s AND column_name = %s",
                         quote_literal_cstr(table_name.c_str()),
                         quote_literal_cstr(column_name.c_str()));
    } else if (!table_name.empty()) {
        appendStringInfo(&buf,
                         "DELETE FROM public.pg_deeplake_metadata WHERE table_name = %s",
                         quote_literal_cstr(table_name.c_str()));
    } else {
        ASSERT(false);
    }
    pg::utils::spi_connector connector;
    if (SPI_execute(buf.data, false, 0) != SPI_OK_DELETE) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to erase metadata")));
    }
}

void save_index_metadata(Oid oid)
{
    pg::utils::memory_context_switcher context_switcher;
    const pg::index_info& idx_info = pg::pg_index::get_index_info(oid);
    StringInfoData buf;
    initStringInfo(&buf);

    appendStringInfo(&buf,
                     "INSERT INTO public.pg_deeplake_metadata (table_name, column_name, index_name, index_type, "
                     "order_type, index_id) "
                     "VALUES (%s, %s, %s, %s, %d, %d) "
                     "ON CONFLICT (table_name, column_name) "
                     "DO UPDATE SET index_name = EXCLUDED.index_name, "
                     "index_type = EXCLUDED.index_type, "
                     "order_type = EXCLUDED.order_type, "
                     "index_id = EXCLUDED.index_id",
                     quote_literal_cstr(idx_info.table_name().c_str()),
                     quote_literal_cstr(idx_info.get_column_names_string().c_str()),
                     quote_literal_cstr(idx_info.index_name().c_str()),
                     quote_literal_cstr(deeplake_core::deeplake_index_type::to_string(idx_info.index_type()).data()),
                     static_cast<int32_t>(idx_info.order_type()),
                     oid);

    pg::utils::spi_connector connector;
    if (SPI_execute(buf.data, false, 0) != SPI_OK_INSERT) {
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to save metadata")));
    }

    // Cross-instance propagation is driven by DDL WAL logging in ProcessUtility.
}

void load_index_metadata()
{
    if (!pg::utils::check_table_exists("pg_deeplake_metadata")) {
        return;
    }
    pg::table_storage::instance().force_load_table_metadata();
    pg::utils::memory_context_switcher context_switcher;

    // Column indices for pg_deeplake_metadata query result (1-based as per SPI API)
    constexpr int COL_TABLE_NAME = 1;
    constexpr int COL_COLUMN_NAME = 2;
    constexpr int COL_INDEX_NAME = 3;
    constexpr int COL_INDEX_TYPE = 4;
    constexpr int COL_ORDER_TYPE = 5;
    constexpr int COL_INDEX_ID = 6;

    const char* query = "SELECT table_name, column_name, index_name, index_type, order_type, index_id "
                        "FROM public.pg_deeplake_metadata";

    pg::utils::spi_connector connector;
    if (SPI_execute(query, true, 0) != SPI_OK_SELECT) {
        base::log_warning(base::log_channel::index, "Failed to query metadata table");
        return;
    }

    const auto proc = SPI_processed;
    const bool res = (proc > 0 && SPI_tuptable != nullptr);
    if (res) {
        TupleDesc tupdesc = SPI_tuptable->tupdesc;
        SPITupleTable* tuptable = SPI_tuptable;

        for (int32_t i = 0; i < proc; ++i) {
            pg::index_info::current().reset();
            HeapTuple tuple = tuptable->vals[i];

            char* val = nullptr;
            if ((val = SPI_getvalue(tuple, tupdesc, COL_TABLE_NAME)) != nullptr) {
                pg::index_info::current().set_table_name(val);
                if (!pg::table_storage::instance().table_exists(val)) {
                    elog(ERROR, "Table %s does not have DeepLake access method, can't create index.", val);
                }
            }
            if ((val = SPI_getvalue(tuple, tupdesc, COL_COLUMN_NAME)) != nullptr) {
                pg::index_info::current().set_column_name(val);
            }
            if ((val = SPI_getvalue(tuple, tupdesc, COL_INDEX_NAME)) != nullptr) {
                pg::index_info::current().set_index_name(val);
            }
            if ((val = SPI_getvalue(tuple, tupdesc, COL_INDEX_TYPE)) != nullptr) {
                pg::index_info::current().set_index_type(val);
            }

            bool is_null = false;
            Datum datum_order_type = SPI_getbinval(tuple, tupdesc, COL_ORDER_TYPE, &is_null);
            const int32_t order_type = is_null ? 0 : DatumGetInt32(datum_order_type);
            pg::index_info::current().set_order_type(static_cast<query_core::order_type>(order_type));

            is_null = false;
            datum_order_type = SPI_getbinval(tuple, tupdesc, COL_INDEX_ID, &is_null);
            const uint32_t oid = is_null ? 0 : DatumGetUInt32(datum_order_type);

            if (!pg::pg_index::has_index_info(oid)) {
                pg::pg_index::load_index_info(oid);
            } else {
                pg::index_info::current().reset();
            }
            context_switcher.reset();
        }
    }
}

/// @}

void deeplake_xact_callback(XactEvent event, void *arg)
{
    switch (event) {
    case XACT_EVENT_PRE_COMMIT:
	case XACT_EVENT_PARALLEL_PRE_COMMIT:
        pg::table_storage::instance().commit_all();
        break;
    case XACT_EVENT_ABORT:
	case XACT_EVENT_PARALLEL_ABORT:
        // Handle transaction abort by rolling back changes
        // Don't throw ERROR here as it would trigger cascading abort loop
        pg::table_storage::instance().rollback_all();
        break;
    default:
        break;
    }
}

void deeplake_subxact_callback(SubXactEvent event,
                               SubTransactionId my_subid,
                               SubTransactionId parent_subid,
                               void* arg)
{
    switch (event) {
    case SUBXACT_EVENT_ABORT_SUB:
        pg::table_storage::instance().rollback_subxact(my_subid);
        break;
    case SUBXACT_EVENT_COMMIT_SUB:
        pg::table_storage::instance().commit_subxact(my_subid, parent_subid);
        break;
    default:
        break;
    }
}

void init_deeplake()
{
    static bool initialized = false;
    if (initialized || !IsUnderPostmaster) {
        return;
    }
    initialized = true;

    // Register exit handler first (runs last due to LIFO order) to use _exit()
    // and avoid C++ static destructor crashes in background workers.
    on_proc_exit(deeplake_quick_exit, 0);

    constexpr int THREAD_POOL_MULTIPLIER = 8;  // Threads per CPU core for async operations
    deeplake_api::initialize(std::make_shared<pg::logger_adapter>(), THREAD_POOL_MULTIPLIER * base::system_report::cpu_cores());

    const std::string redis_url = base::getenv<std::string>("REDIS_URL", "");
    if (!redis_url.empty()) {
        deeplake_api::initialize_redis_cache(redis_url, 86400,
                                             deeplake_api::metadata_catalog_cache_pattern);
    }

    pg::table_storage::instance(); /// initialize table storage

    RegisterXactCallback(deeplake_xact_callback, nullptr);
    RegisterSubXactCallback(deeplake_subxact_callback, nullptr);
}

} // namespace pg
