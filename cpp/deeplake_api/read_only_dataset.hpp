#pragma once

#include "branches.hpp"
#include "history.hpp"
#include "prefetcher.hpp"
#include "schema.hpp"
#include "tags.hpp"
#include "tags_view.hpp"
#include "types.hpp"

#include <deeplake/column_datafiles_info.hpp>

#include <heimdall/dataset_view.hpp>
#include <icm/string_set.hpp>

#include <memory>

namespace http {
class uri;
} // namespace http

namespace deeplake_api {

class read_only_dataset : public heimdall::dataset_view
{
public:
    [[nodiscard]] const std::string& id() const;
    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] const std::string& description() const;
    [[nodiscard]] const icm::json& metadata() const override;
    [[nodiscard]] std::chrono::system_clock::time_point created_time() const;
    [[nodiscard]] branch_view get_current_branch() const;
    [[nodiscard]] branches_view get_branches() const;
    [[nodiscard]] std::shared_ptr<tags_view> get_tags() const;
    [[nodiscard]] std::shared_ptr<deeplake_api::history> history() const;
    [[nodiscard]] async::promise<void> checkout(const std::string& string);
    [[nodiscard]] std::shared_ptr<heimdall::schema_view> get_schema_view() const override;
    [[nodiscard]] std::shared_ptr<schema> get_schema() const;
    [[nodiscard]] heimdall::column_id_t get_column_id(const std::string& name) const;
    [[nodiscard]] int64_t num_rows() const override;
    [[nodiscard]] int columns_count() const override;
    [[nodiscard]] heimdall::column_view& get_column_view(int index) override;
    [[nodiscard]] const std::optional<std::string>& get_creds_key() const;
    [[nodiscard]] const std::string& version() const;
    [[nodiscard]] async::promise<void> refresh();
    [[nodiscard]] const std::shared_ptr<storage::reader>& get_reader() const;
    [[nodiscard]] async::promise<void> push(const std::string& url,
                                            icm::string_map<>&& params,
                                            const std::optional<std::string>& token = std::nullopt) const;
    [[nodiscard]] async::promise<void> push(const std::shared_ptr<storage::writer>& writer) const;

    [[nodiscard]] std::vector<std::pair<std::string, int64_t>> get_datafiles() const;
    [[nodiscard]] std::map<std::string, deeplake::column_datafiles_info> get_datafiles_report() const;

public:
    read_only_dataset(std::shared_ptr<deeplake::dataset_view> dataset,
                      std::shared_ptr<deeplake::data_container_view> data_container);
    [[nodiscard]] icm::json to_json() const override;
    [[nodiscard]] static async::promise<std::shared_ptr<read_only_dataset>> from_json(const icm::const_json& json);

private:
    std::shared_ptr<deeplake::dataset_view> dataset_;
    std::shared_ptr<deeplake::data_container_view> data_container_;
};

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>>
open_read_only(const std::string& url, icm::string_map<>&& params, std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>> open_read_only(const http::uri& url,
                                                                                icm::string_map<>&& params);

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>> open_read_only(const branch_view& b);

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>>
open_read_only(const std::shared_ptr<storage::reader>& storage);

inline std::string get_read_only_dataset_json(const std::shared_ptr<deeplake_api::read_only_dataset>& ds)
{
    return async::run_on_main([ds]() {
               auto json = ds->to_json();
               json["type"] = "read_only_dataset";
               return json.dump();
           })
        .get_future()
        .get();
}

} // namespace deeplake_api
