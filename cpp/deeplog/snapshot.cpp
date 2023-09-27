#include "snapshot.hpp"

#include <utility>

namespace deeplog {

    snapshot::snapshot(std::string branch_id, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            base_snapshot(deeplog),
            branch_id(std::move(branch_id)) {}

    snapshot::snapshot(std::string branch_id, const long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            base_snapshot(version, deeplog),
            branch_id(std::move(branch_id)) {}

    std::vector<std::shared_ptr<add_file_action>> snapshot::data_files() {
        std::vector<std::shared_ptr<add_file_action>> actions = {};

        for (auto it = find_actions(typeid(add_file_action)); it != actions_->end(); ++it) {
            actions.push_back(std::dynamic_pointer_cast<add_file_action>(*it));
        }

        return actions;
    }

    std::vector<std::shared_ptr<create_commit_action>> snapshot::commits() {
        std::vector<std::shared_ptr<create_commit_action>> actions = {};

        for (auto it = find_actions(typeid(create_commit_action)); it != actions_->end(); ++it) {
            actions.push_back(std::dynamic_pointer_cast<create_commit_action>(*it));
        }

        return actions;
    }

    std::vector<std::shared_ptr<create_tensor_action>> snapshot::tensors() {
        std::vector<std::shared_ptr<create_tensor_action>> actions = {};

        for (auto it = find_actions(typeid(create_tensor_action)); it != actions_->end(); ++it) {
            actions.push_back(std::dynamic_pointer_cast<create_tensor_action>(*it));
        }

        return actions;
    }

}