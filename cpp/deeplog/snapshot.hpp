#pragma once

#include "deeplog.hpp"
#include "actions/create_commit_action.hpp"
#include "base_snapshot.hpp"

namespace deeplog {

    class deeplog;

    class snapshot : public base_snapshot {

    public:

        snapshot(std::string branch_id, const std::shared_ptr<::deeplog::deeplog> &deeplog);

        snapshot(std::string branch_id, const unsigned long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog);

        std::vector<std::shared_ptr<add_file_action>> data_files();

        std::vector<std::shared_ptr<create_tensor_action>> tensors();

        std::vector<std::shared_ptr<create_commit_action>> commits();
    };
}