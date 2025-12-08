#pragma once

/**
 * @defgroup tql
 * @{
 * @brief Tensor Query Language - Query engine of Deeplake.
 *
 * @}
 */

#include "explain_query_result.hpp"
#include "options.hpp"
#include "parsing_result.hpp"

#include <icm/string_map.hpp>
#include <query_core/top_k_search_info.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace async {
template <typename T>
class promise;
} // namespace async

namespace heimdall {
class dataset_view;
}

namespace nd {
class array;
}

namespace tql {

class executor;

const auto CHANNEL_TQL = "deeplake.tql";

/**
 * @brief Runs the given query and returns the result dataset view.
 *
 * @param query_string TQL query string,
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 * @return Promise to the result dataset.
 */
[[nodiscard]] async::promise<heimdall::dataset_view_ptr> query(const std::string& query_string,
                                                               options opts = options::default_config(),
                                                               const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Runs the given query on the given dataset and returns the result dataset view.
 *
 * @param query_string TQL query string,
 * @param dataset Input dataset.
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 * @return Promise to the result dataset.
 */
[[nodiscard]] async::promise<heimdall::dataset_view_ptr> query(const std::string& query_string,
                                                               heimdall::dataset_view_ptr dataset,
                                                               options opts = options::default_config(),
                                                               const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Parse the given query and returns the executor.
 *
 * @param query_string TQL query string,
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 * @return Executor to the result dataset.
 */
[[nodiscard]] async::promise<std::shared_ptr<executor>>
prepare_query(const std::string& query_string,
              options opts = options::default_config(),
              const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Parse the given query on the given dataset and returns the executor.
 *
 * @param query_string TQL query string,
 * @param dataset Input dataset.
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 * @return Executor to the result dataset.
 */
[[nodiscard]] async::promise<std::shared_ptr<executor>>
prepare_query(const std::string& query_string,
              heimdall::dataset_view_ptr dataset,
              options opts = options::default_config(),
              const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Explains the given query and returns the string containing the explanation.
 *
 * @param query_string TQL query string.
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 */
[[nodiscard]] async::promise<explain_query_result> explain_query(const std::string& query_string,
                                                                 options opts = options::default_config(),
                                                                 const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Explains the given query on the given dataset and returns the string containing the explanation.
 *
 * @param query_string TQL query string.
 * @param dataset Input dataset.
 * @param opts Options to control query engine behavior.
 * @param params Parameters which are passed to data-source registry to get source dataset. Parameters can be any info
 * which source registry is used to access data, such as tokens credentials etc.
 */
[[nodiscard]] async::promise<explain_query_result> explain_query(const std::string& query_string,
                                                                 heimdall::dataset_view_ptr dataset,
                                                                 options opts = options::default_config(),
                                                                 const icm::string_map<>& params = icm::string_map<>{});

/**
 * @brief Parses the given query string and returns the result containing ds url and tensors.
 *
 * @param query_string TQL query string.
 * @param opts Options to control query engine behavior.
 * @return parsing_result.
 */
std::vector<std::shared_ptr<parsing_result>> parse_query(const std::string& query_string,
                                                         options opts = options::default_config());

/**
 * @brief Returns keywords list of the query language.
 *
 * @return std::vector<std::string> List of keywords.
 */
std::vector<std::string> query_keywords();

/**
 * @brief Checks if the given query string involves only filtering or not.
 * Filtering is the query which only slices the dataset without involving
 * operations which would change the structure - group, ungroup, tensor_indexing.
 *
 * @param query_string Query string.
 * @return true If the given query is filtering.
 * @return false If the given query is not filtering.
 */
bool is_filter(const std::string& query_string);

/**
 * @brief Checks if the given query string is valid.
 *
 * @param query_string
 * @return true If the query_string if valid query.
 * @return false If the query_string is not valid query.
 */
bool is_valid(const std::string& query_string);

/**
 * @brief Checks if the given query is searches for top k elements and the order is binary function.
 *
 * @param query_string
 * @return true If the query_string if valid query.
 * @return false If the query_string is not valid query.
 */
query_core::top_k_binary_function_search_info check_for_top_k_binary_function_search(const std::string& query_string);

/**
 * @brief Registers the given custom nullary function to be used in TQL.
 *
 * @param name Function name.
 * @param function Function
 */
void register_function(std::string name, std::function<nd::array()> function);

/**
 * @brief Registers the given custom unary function to be used in TQL.
 *
 * @param name Function name.
 * @param function Function
 */
void register_function(std::string name, std::function<nd::array(const nd::array&)> function, bool is_batch = false);

/**
 * @brief Registers the given custom binary function to be used in TQL.
 *
 * @param name Function name.
 * @param function Function
 */
void register_function(std::string name,
                       std::function<nd::array(const nd::array&, const nd::array&)> function,
                       bool is_batch = false);

/**
 * @brief Unregisters the given custom function.
 *
 * @param name Function name.
 */
void unregister_function(std::string name);

/**
 * @brief Checks if tql has user defined function with the given name.
 *
 * @param name Function name.
 */
bool has_registered_function(const std::string& name);

/**
 * @brief Initializes TQL.
 */
void initialize();

void deinitialize();

} // namespace tql
