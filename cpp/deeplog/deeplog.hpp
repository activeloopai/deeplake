#pragma once

#include <string>
#include <arrow/status.h>
#include "actions/add_file_action.hpp"
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"
#include "actions/create_tensor_action.hpp"
#include "actions/create_commit_action.hpp"

namespace deeplog {

    const std::string MAIN_BRANCH_ID = "";
    const std::string MAIN_BRANCH_NAME = "main";

    template<typename T>
    struct deeplog_state {
        deeplog_state(T data, long version) : data(data), version(version) {}

        long version;
        T data;
    };

    class deeplog {
    public:
        [[nodiscard]] static std::shared_ptr<deeplog> create(const std::string &path);

        [[nodiscard]] static std::shared_ptr<deeplog> open(const std::string &path);

        const std::string path;

        virtual int log_format() const;

        long version(const std::string &branch_id) const;

        deeplog_state<std::shared_ptr<protocol_action>> protocol() const;

        deeplog_state<std::shared_ptr<metadata_action>> metadata() const;

        deeplog_state<std::vector<std::shared_ptr<create_branch_action>>> branches() const;

        deeplog_state<std::shared_ptr<create_branch_action>> branch_by_id(const std::string &branch_id) const;

        deeplog_state<std::vector<std::shared_ptr<create_tensor_action>>> tensors(const std::string &branch_id, const std::optional<long> &version) const;

        deeplog_state<std::vector<std::shared_ptr<add_file_action>>> data_files(const std::string &branch_id, const std::optional<long> &version);

        deeplog_state<std::vector<std::shared_ptr<create_commit_action>>> commits(const std::string &branch_id, const std::optional<long> &version);

        void commit(const std::string &branch_id,
                    const long &base_version,
                    const std::vector<std::shared_ptr<action>> &actions);

        void checkpoint(const std::string &branch_id);

        arrow::Result<std::shared_ptr<arrow::Table>> operations(const std::string &branch_id, const long &from, const std::optional<long> &to) const;
        deeplog_state<std::vector<std::shared_ptr<action>>> list_actions(const std::string &branch_id, const long &from, const std::optional<long> &to) const;

    private:

        //only created through open() etc.
        deeplog(std::string path);

        arrow::Result<std::shared_ptr<arrow::Table>> read_checkpoint(const std::string &dir_path, const long &version) const;
        arrow::Result<std::shared_ptr<arrow::StructScalar>> last_value(const std::string &column_name, const std::shared_ptr<arrow::Table> &table) const;
        arrow::Status write_checkpoint(const std::string &branch_id, const long &version);

        long file_version(const std::filesystem::path &path) const;

        const static std::shared_ptr<arrow::Schema> arrow_schema;
    };

}
