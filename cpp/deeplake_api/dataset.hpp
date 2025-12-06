#pragma once

#include "branches.hpp"
#include "exceptions.hpp"
#include "history.hpp"
#include "read_only_dataset.hpp"
#include "schema.hpp"
#include "tag.hpp"
#include "tags.hpp"
#include "types.hpp"

#include <async/promise.hpp>
#include <async/queue.hpp>
#include <async/timer_manager.hpp>
#include <deeplake/column_datafiles_info.hpp>
#include <deeplake/types.hpp>
#include <heimdall/dataset.hpp>

#include <memory>

namespace deeplake {
class dataset;
class data_container;
} // namespace deeplake

namespace deeplake_api {

class dataset : public heimdall::dataset
{
public:
    [[nodiscard]] async::promise<bool> is_auto_commit_enabled() const;
    [[nodiscard]] async::promise<void> set_auto_commit_enabled(bool enabled);

    const std::string& id() const;
    const std::string& name() const;
    void set_name(const std::string& name);
    const std::string& description() const;
    void set_description(const std::string& description);
    void set_metadata(const std::string& key, const icm::json& value);
    const icm::json& metadata() const override;
    std::chrono::system_clock::time_point created_time() const;
    deeplake::indexing_mode get_indexing_mode() const;
    void set_indexing_mode(deeplake::indexing_mode mode);
    branch get_current_branch() const;
    branches get_branches() const;
    std::shared_ptr<tags> get_tags() const;

    [[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>> lock();

    [[nodiscard]] async::promise<branch> create_branch(const std::string& name);
    [[nodiscard]] async::promise<branch> create_branch(const std::string& name, const std::string& version);

    [[nodiscard]] async::promise<void>
    merge(const std::string& branch_name, std::optional<std::string> version, std::optional<std::string> message);

    [[nodiscard]] async::promise<deeplake_api::tag> tag(const std::string& name, const std::string& message);
    [[nodiscard]] async::promise<deeplake_api::tag>
    tag(const std::string& name, const std::string& version, const std::string& message);
    [[nodiscard]] async::promise<deeplake_api::tag> tag(const std::string& name,
                                                        const std::string& version,
                                                        const std::string& message,
                                                        const std::string& query_string);
    [[nodiscard]] async::promise<deeplake_api::tag> tag(const std::string& name,
                                                        const std::string& version,
                                                        const std::string& message,
                                                        const icm::index_mapping_t<int64_t>& rows);

    /**
     * @brief Tries to find the source dataset of the given view and create tag on the source dataset.
     * @param name the name of the tag
     * @param dataset_view the view to tag
     */
    [[nodiscard]] static async::promise<deeplake_api::tag>
    tag_view(const std::string& name, const std::string& message, const heimdall::dataset_view_ptr& dataset_view);

    [[nodiscard]] async::promise<void> push(const std::string& url,
                                            icm::string_map<>&& params,
                                            const std::optional<std::string>& token = std::nullopt) const;

    [[nodiscard]] async::promise<void> push(const std::shared_ptr<storage::writer>& writer) const;

    [[nodiscard]] async::promise<void> pull(const std::shared_ptr<storage::reader>& reader);

    [[nodiscard]] async::promise<void>
    pull(const std::string& url, icm::string_map<>&& params, const std::optional<std::string>& token = std::nullopt);

    /**
     * @brief Rebuild a branch from scratch to remediate integrity issues.
     * This is an internal dangerous operation that creates a clean copy of the branch.
     * Warning: This changes the branch ID and may change branch parent relationships.
     * @param branch_id_or_name The branch to rebuild
     * @return Promise that resolves when the rebuild is complete
     */
    [[nodiscard]] async::promise<void> rebuild_branch(const std::string& branch_id_or_name);

    /**
     * @brief Check integrity and return formatted report
     * @return String containing integrity report
     */
    [[nodiscard]] std::string check_integrity() const;

    [[nodiscard]] std::shared_ptr<class history> history() const;
    std::shared_ptr<heimdall::schema_view> get_schema_view() const override;
    std::shared_ptr<schema> get_schema() const;
    heimdall::column_id_t get_column_id(const std::string& name) const;
    int64_t num_rows() const override;
    int columns_count() const override;
    heimdall::column& get_column(int index) override;
    heimdall::column& get_column(int index) const;
    heimdall::column& get_column(std::string_view name);
    const heimdall::column& get_column(std::string_view name) const;

    void add_column(const std::string& name,
                    const nd::type& tp,
                    const std::optional<nd::array>& default_value = std::nullopt);

    void
    add_column(const std::string& name, const type& tp, const std::optional<nd::array>& default_value = std::nullopt);

    void remove_column(std::string name);
    void rename_column(const std::string& name, std::string new_name);
    [[nodiscard]] async::promise<void> append_row(const icm::string_map<nd::array>& row) override;
    [[nodiscard]] async::promise<void> append_rows(const icm::string_map<nd::array>& rows) override;
    [[nodiscard]] async::promise<void>
    update_row(int64_t row_id, const std::string& column_name, const nd::array& new_value) override;
    [[nodiscard]] async::promise<void> update_rows(int64_t start_row_id,
                                                   int64_t end_row_id,
                                                   const std::string& column_name,
                                                   const nd::array& new_value) override;
    [[nodiscard]] async::promise<void> update_row(int64_t row_id,
                                                  const icm::string_map<nd::array>& new_values) override;
    [[nodiscard]] async::promise<void>
    update_rows(int64_t start_row_id, int64_t end_row_id, const icm::string_map<nd::array>& new_values) override;
    void delete_row(int64_t row_id) override;
    void delete_rows(const std::vector<int64_t>& row_ids) override;
    void delete_rows(int64_t start_row_id, int64_t end_row_id) override;

    /**
     * @return the current version of the container
     */
    [[nodiscard]] std::string version() const;

    /**
     * Commits the changes in this container.
     */
    [[nodiscard]] async::promise<void> commit(const std::optional<std::string>& message = std::nullopt);

    /**
     * Refreshes the container to be based on the latest version in the branch.
     */
    [[nodiscard]] async::promise<void> refresh();

    const std::shared_ptr<storage::reader>& get_reader() const;
    const std::shared_ptr<storage::writer>& get_writer() const;
    [[nodiscard]] async::promise<void> set_creds_key(const std::string& key,
                                                     const std::optional<std::string>& token = std::nullopt);
    const std::optional<std::string>& get_creds_key() const;
    std::vector<std::pair<std::string, int64_t>> get_datafiles() const;
    std::map<std::string, deeplake::column_datafiles_info> get_datafiles_report() const;

public:
    static std::shared_ptr<dataset>
    make_dataset(std::shared_ptr<deeplake::dataset> ds,
                 std::shared_ptr<deeplake::data_container> dc,
                 base::function<void(dataset*)> deleter = std::default_delete<dataset>());

    ~dataset() noexcept override;

    bool has_uncommitted_changes() const;

    [[nodiscard]] icm::json to_json() const override;

    [[nodiscard]] static async::promise<std::shared_ptr<dataset>>
    from_json(const icm::const_json& json, base::function<void(dataset*)> deleter = std::default_delete<dataset>());

private:
    auto shared_self()
    {
        return std::static_pointer_cast<dataset>(shared_from_this());
    }

    auto shared_self() const
    {
        return std::static_pointer_cast<const dataset>(shared_from_this());
    }

    dataset(std::shared_ptr<deeplake::dataset> ds, std::shared_ptr<deeplake::data_container> dc);
    void setup_auto_commit();

private:
    std::shared_ptr<deeplake::dataset> dataset_;
    std::shared_ptr<deeplake::data_container> data_container_;

    async::timer_manager::timer_id_t auto_commit_timer_id_ = 0;
    std::chrono::steady_clock::time_point last_commit_time_;

    std::string session_id_;
    std::unique_ptr<async::bg_queue> log_queue_;
    std::vector<async::promise<void>> pending_log_writes_;
    void log_operation(const std::string& operation_name, const icm::json& args);

public:
    void start_logging();
    void stop_logging();
    bool is_logging_enabled() const
    {
        return static_cast<bool>(log_queue_);
    }
};

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create(const std::string& url,
       icm::string_map<>&& creds,
       std::optional<std::string> token = std::nullopt,
       bool start_logging = false,
       base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create(const http::uri& url,
       icm::string_map<>&& creds,
       bool start_logging = false,
       base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create(const std::shared_ptr<storage::writer>& storage,
       bool start_logging = false,
       base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create_with_schema(const std::string& url,
                   icm::string_map<deeplake_core::type>&& schema,
                   icm::string_map<>&& creds,
                   std::optional<std::string> token = std::nullopt,
                   bool start_logging = false,
                   base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create_with_schema(const http::uri& url,
                   icm::string_map<deeplake_core::type>&& schema,
                   icm::string_map<>&& creds,
                   bool start_logging = false,
                   base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
create_with_schema(const std::shared_ptr<storage::writer>& storage,
                   icm::string_map<deeplake_core::type>&& schema,
                   bool start_logging = false,
                   base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>> link(const std::string& source_url,
                                                                      const std::string& destination_url,
                                                                      icm::string_map<>&& creds,
                                                                      std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>>
link(const std::string& source_url, const http::uri& destination_url, icm::string_map<>&& creds);

[[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>>
link(const std::string& source_url, const std::shared_ptr<storage::writer>& destination_storage);

[[nodiscard]] async::promise<void>
delete_dataset(const std::string& url, icm::string_map<>&& creds, std::optional<std::string> token = std::nullopt);
[[nodiscard]] async::promise<void> delete_dataset(const http::uri& url, icm::string_map<>&& creds);
[[nodiscard]] async::promise<void> delete_dataset(const std::shared_ptr<storage::writer>& storage);

[[nodiscard]] async::promise<void> copy(const std::string& source,
                                        const std::string& dst,
                                        icm::string_map<>&& src_creds = {},
                                        icm::string_map<>&& dst_creds = {},
                                        std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<void> copy(const http::uri& source,
                                        const http::uri& dst,
                                        icm::string_map<>&& src_creds = {},
                                        icm::string_map<>&& dst_creds = {},
                                        std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<void> copy(const std::shared_ptr<storage::reader>& src,
                                        const std::shared_ptr<storage::writer>& dst);

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
open(const std::string& url,
     icm::string_map<>&& params,
     std::optional<std::string> token = std::nullopt,
     base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
open(const http::uri& url, icm::string_map<>&& params, base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<std::shared_ptr<dataset>> open(const branch& b);

[[nodiscard]] async::promise<std::shared_ptr<dataset>>
open(const std::shared_ptr<storage::writer>& storage, base::function<void(dataset*)> = std::default_delete<dataset>());

[[nodiscard]] async::promise<bool>
exists(const std::string& url, icm::string_map<>&& params, std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<bool> exists(const std::shared_ptr<storage::reader>& storage);

[[nodiscard]] async::promise<bool> exists(const http::uri& url, icm::string_map<>&& params);

[[nodiscard]] async::promise<std::shared_ptr<dataset>> connect(const std::string& src_url,
                                                               const std::string& org_id,
                                                               std::optional<std::string> ds_name,
                                                               std::optional<std::string> creds_key,
                                                               std::optional<std::string> token = std::nullopt);

[[nodiscard]] async::promise<void> disconnect(const std::string& url, std::optional<std::string> token = std::nullopt);

[[nodiscard]] inline auto set_dataset_name(const std::shared_ptr<dataset>& ds, const std::string& name)
{
    return async::run_on_main([ds, name]() {
        ds->set_name(name);
    });
}

[[nodiscard]] inline auto set_dataset_description(const std::shared_ptr<dataset>& ds, const std::string& description)
{
    return async::run_on_main([ds, description]() {
        ds->set_description(description);
    });
}

[[nodiscard]] inline auto add_column(const std::shared_ptr<dataset>& ds,
                                     const std::string& name,
                                     const type& tp,
                                     const std::optional<nd::array>& default_value = std::nullopt)
{
    return async::run_on_main([ds, name, tp, default_value]() {
        ds->add_column(name, tp, default_value);
    });
}

[[nodiscard]] inline auto remove_column(const std::shared_ptr<dataset>& ds, const std::string& name)
{
    return async::run_on_main([ds, name]() mutable {
        ds->remove_column(std::move(name));
    });
}

[[nodiscard]] inline auto
rename_column(const std::shared_ptr<dataset>& ds, const std::string& name, std::string new_name)
{
    return async::run_on_main([ds, name, new_name = std::move(new_name)]() mutable {
        ds->rename_column(name, std::move(new_name));
    });
}

[[nodiscard]] inline auto delete_row(const std::shared_ptr<dataset>& ds, int64_t row_id)
{
    return async::run_on_main([ds, row_id]() {
        ds->delete_row(row_id);
    });
}

[[nodiscard]] inline auto delete_rows(const std::shared_ptr<dataset>& ds, const std::vector<int64_t>& row_ids)
{
    return async::run_on_main([ds, row_ids]() {
        ds->delete_rows(row_ids);
    });
}

[[nodiscard]] inline auto delete_rows(const std::shared_ptr<dataset>& ds, int64_t start_row_id, int64_t end_row_id)
{
    return async::run_on_main([ds, start_row_id, end_row_id]() {
        ds->delete_rows(start_row_id, end_row_id);
    });
}

[[nodiscard]] inline auto merge(const std::shared_ptr<dataset>& ds,
                                const std::string& branch_name,
                                const std::optional<std::string>& version,
                                const std::optional<std::string>& message)
{
    return async::run_on_main([ds, branch_name, version, message]() {
        return ds->merge(branch_name, version, message);
    });
}

[[nodiscard]] inline auto dataset_tag(const std::shared_ptr<dataset>& ds,
                                      const std::string& name,
                                      const std::optional<std::string>& message,
                                      const std::optional<std::string>& version)
{
    return async::run_on_main([ds, name, message, version]() {
        if (version.has_value()) {
            return ds->tag(name, version.value(), message.value_or(""));
        }
        return ds->tag(name, message.value_or(""));
    });
}

} // namespace deeplake_api
