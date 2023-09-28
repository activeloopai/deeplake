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

    class deeplog {
    public:
        [[nodiscard]] static std::shared_ptr<deeplog> create(const std::string &path, const int &log_version);

        [[nodiscard]] static std::shared_ptr<deeplog> open(const std::string &path);

        const std::string path;

        virtual int log_format() const;

        long version(const std::string &branch_id) const;

        std::vector<std::shared_ptr<create_commit_action>> commits(const std::string &branch_id, const std::optional<long> &version);

        void commit(const std::string &branch_id,
                    const long &base_version,
                    const std::vector<std::shared_ptr<action>> &actions);

        void checkpoint(const std::string &branch_id);

        arrow::Result<std::shared_ptr<arrow::Table>> action_data(const std::string &branch_id, const long &from, const std::optional<long> &to) const;

        std::tuple<std::shared_ptr<std::vector<std::shared_ptr<action>>>, long> get_actions(const std::string &branch_id, const std::optional<long> &to) const;

    protected:

        //only created through open() etc.
        deeplog(std::string path);

    private:

        arrow::Result<std::shared_ptr<arrow::Table>> read_checkpoint(const std::string &dir_path, const long &version) const;

        arrow::Status write_checkpoint(const std::string &branch_id, const long &version);

        long file_version(const std::filesystem::path &path) const;

        const static std::shared_ptr<arrow::Schema> arrow_schema;
    };

}
