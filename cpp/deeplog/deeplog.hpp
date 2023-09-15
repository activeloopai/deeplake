#ifndef DEEPLAKE_DEEPLOG_HPP
#define DEEPLAKE_DEEPLOG_HPP

#include <string>
#include <arrow/status.h>
#include "actions/add_file_action.hpp"
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"

namespace deeplake {

    const std::string MAIN_BRANCH_ID = "";

    template<typename T>
    struct deeplog_state {
        deeplog_state(T data, long version) : data(data), version(version) {}

        long version;
        T data;
    };

    class deeplog {
    public:
        static std::shared_ptr<deeplog> create(const std::string &path);

        static std::shared_ptr<deeplog> open(const std::string &path);

        std::string path();

        long version() const;

        long version(const std::string &branch_id) const;

        deeplog_state<std::shared_ptr<deeplake::protocol_action>> protocol() const;

        deeplog_state<std::shared_ptr<deeplake::metadata_action>> metadata() const;

        deeplog_state<std::vector<std::shared_ptr<deeplake::create_branch_action>>> branches() const;

        deeplog_state<std::shared_ptr<deeplake::create_branch_action>> branch_by_id(const std::string &branch_id) const;

        deeplog_state<std::vector<std::shared_ptr<deeplake::add_file_action>>> data_files(const std::string &branch_id, const std::optional<long> &version);

        void commit(const std::string &branch_id,
                    const long &base_version,
                    const std::vector<deeplake::action *> &actions);

        void checkpoint(const std::string &branch_id);

    private:

        //only created through open() etc.
        deeplog(std::string path);

        deeplog_state<std::vector<std::shared_ptr<action>>> list_actions(const std::string &branch_id, const long &from, const std::optional<long> &to) const;

        arrow::Status read_checkpoint(const std::string &dir_path, const long &version, std::vector<std::shared_ptr<action>> &actions) const;

        arrow::Status write_checkpoint(const std::string &branch_id, const long &version);

        long file_version(const std::filesystem::path &path) const;

        std::string path_;
    };

} // deeplake

#endif //DEEPLAKE_DEEPLOG_HPP
