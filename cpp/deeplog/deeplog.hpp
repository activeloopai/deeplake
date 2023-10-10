#pragma once

#include <fstream>
#include <arrow/api.h>
#include "actions/add_file_action.hpp"
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"
#include "actions/create_tensor_action.hpp"
#include "actions/create_commit_action.hpp"
#include "../storage/storage.hpp"

namespace deeplog {

    const std::string META_BRANCH_ID = "_meta";

    std::string zero_pad(const unsigned long &version);

    class deeplog {
    public:
        [[nodiscard]] static std::shared_ptr<deeplog> create(const std::shared_ptr<storage::storage> &storage, const int &log_version);

        [[nodiscard]] static std::shared_ptr<deeplog> create(const std::string &path, const int &log_version);

        [[nodiscard]] static std::shared_ptr<deeplog> open(const std::string &path);

        [[nodiscard]] static std::shared_ptr<deeplog> open(const std::shared_ptr<storage::storage> &storage);

        virtual int log_format() const;

        unsigned long version(const std::string &branch_id) const;

        std::vector<std::shared_ptr<create_commit_action>> commits(const std::string &branch_id, const std::optional<unsigned long> &version);

        void commit(const std::string &branch_id,
                    const unsigned long &base_version,
                    const std::vector<std::shared_ptr<action>> &actions);

        void checkpoint(const std::string &branch_id);

        arrow::Result<std::shared_ptr<arrow::Table>> action_data(const std::string &branch_id, const unsigned long &from, const std::optional<unsigned long> &to) const;

        std::tuple<std::shared_ptr<std::vector<std::shared_ptr<action>>>, long> get_actions(const std::string &branch_id, const std::optional<unsigned long> &to) const;

    protected:

        //only created through open() etc.
        deeplog(const std::shared_ptr<storage::storage> &storage);

    private:

        arrow::Result<std::shared_ptr<arrow::Table>> read_checkpoint(const std::string &dir_path, const unsigned long &version) const;

        arrow::Status write_checkpoint(const std::string &branch_id, const unsigned long &version);

        long file_version(const std::string &path) const;

        const static std::shared_ptr<arrow::Schema> arrow_schema;

        std::shared_ptr<storage::storage> storage_;

        std::shared_ptr<arrow::io::RandomAccessFile> open_arrow_istream(const storage::file_ref &file) const;

        std::vector<std::shared_ptr<arrow::ArrayBuilder>> create_arrow_builders() const;
    };

}
