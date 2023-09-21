#include "snapshot.hpp"

#include <utility>

namespace deeplog {

    snapshot::snapshot(std::string branch_id, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            branch_id(std::move(branch_id)),
            version(deeplog->version(branch_id)),
            deeplog(deeplog){
    }

    snapshot::snapshot(std::string branch_id, const long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            branch_id(std::move(branch_id)),
            version(version),
            deeplog(deeplog){
    }

    std::vector<std::shared_ptr<add_file_action>> snapshot::data_files() {
        return deeplog->data_files(branch_id, version).data;
    }

    std::vector<std::shared_ptr<create_commit_action>> snapshot::commits() {
        return deeplog->commits(branch_id, version).data;
    }

    std::vector<std::shared_ptr<create_tensor_action>> snapshot::tensors() {
        return deeplog->tensors(branch_id, version).data;
    }

}