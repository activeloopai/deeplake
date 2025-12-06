#pragma once

#include "explain_query_result.hpp"
#include "options.hpp"
#include "table.hpp"

#include <async/promise.hpp>
#include <async/shared_promise.hpp>
#include <heimdall/dataset_view.hpp>
#include <icm/string_map.hpp>
#include <nd/array.hpp>
#include <query_core/statement.hpp>

#include <memory>

namespace hsql {
class OrderDescription;
class SelectStatement;
class SetOperation;
} // namespace hsql

namespace tql {

class from_expression;

class executor : public std::enable_shared_from_this<executor>
{
public:
    executor(const std::string& query_string,
             const hsql::SelectStatement& statement,
             heimdall::dataset_view_ptr dataset,
             std::shared_ptr<void> context,
             tql::options opts,
             const icm::string_map<>& params)
        : statement_(statement)
        , query_string_(query_string)
        , dataset_(dataset)
        , context_(std::move(context))
        , options_(opts)
        , params_(params)
    {}

    inline auto get_query_string() const noexcept
    {
        return query_string_;
    }

    async::promise<heimdall::dataset_view_ptr> run();
    async::promise<std::vector<heimdall::dataset_view_ptr>> run_batch(nd::array params);
    async::promise<explain_query_result> explain_query();

private:
    async::promise<heimdall::dataset_view_ptr> run_internal();
    async::promise<std::vector<heimdall::dataset_view_ptr>> run_batch_internal(nd::array params);

    struct single_query_result
    {
        single_query_result(async::promise<table> source,
                            const hsql::SelectStatement* statement,
                            const hsql::SetOperation* op)
            : source_(std::move(source))
            , statement_(statement)
            , op_(op)
        {
        }

        async::shared_promise<table> source_;
        const hsql::SelectStatement* statement_;
        const hsql::SetOperation* op_;
    };

    async::promise<std::vector<std::vector<heimdall::dataset_view_ptr>>>
    collect_datasets(std::vector<executor::single_query_result>& ss,
                     nd::array current_params);

    async::promise<heimdall::dataset_view_ptr>
    combine_datasets(std::vector<heimdall::dataset_view_ptr> datasets,
                     const std::vector<executor::single_query_result>& ss,
                     nd::array current_params);

    async::promise<tql::table> get_data_source(const tql::from_expression& from);

    void get_all_statements(const hsql::SelectStatement& statement, const hsql::SetOperation* op,
                            std::vector<executor::single_query_result>& output);
    async::promise<heimdall::dataset_view_ptr> merge_datasets(std::vector<heimdall::dataset_view_ptr> datasets,
                                                              const std::vector<const hsql::SetOperation*>& operations,
                                                              nd::array current_params);
    async::promise<heimdall::dataset_view_ptr>
    merge_datasets(heimdall::dataset_view_ptr ds1, heimdall::dataset_view_ptr ds2, const hsql::SetOperation* op, nd::array current_params);
    heimdall::dataset_view_ptr
    union_(heimdall::dataset_view_ptr ds1, heimdall::dataset_view_ptr ds2, query_core::limit_statement l);
    heimdall::dataset_view_ptr
    intersect_(heimdall::dataset_view_ptr ds1, heimdall::dataset_view_ptr ds2, query_core::limit_statement l);
    heimdall::dataset_view_ptr
    difference_(heimdall::dataset_view_ptr ds1, heimdall::dataset_view_ptr ds2, query_core::limit_statement l);
    async::promise<heimdall::dataset_view_ptr> apply_order(heimdall::dataset_view_ptr ds,
                                                           const std::vector<hsql::OrderDescription*>* order,
                                                           nd::array current_params);

private:
    const hsql::SelectStatement& statement_;
    std::string query_string_;
    heimdall::dataset_view_ptr dataset_;
    std::shared_ptr<void> context_;
    tql::options options_;
    icm::string_map<> params_;
    std::map<std::pair<std::string, icm::string_map<>>, heimdall::dataset_view_ptr> query_cache_;
};

int32_t get_max_num_parallel_queries();
void set_max_num_parallel_queries(int32_t max_num_parallel_queries);

} // namespace tql
