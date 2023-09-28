#include "metadata_snapshot.hpp"

namespace deeplog {
    metadata_snapshot::metadata_snapshot(const std::shared_ptr<::deeplog::deeplog> &deeplog) : base_snapshot(deeplog) {}

    metadata_snapshot::metadata_snapshot(const long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) : base_snapshot(version, deeplog) {}

    std::shared_ptr<protocol_action> metadata_snapshot::protocol() const {
        return std::dynamic_pointer_cast<protocol_action>(find_action(typeid(protocol_action)));
    }

    std::shared_ptr<metadata_action> metadata_snapshot::metadata() const {
        return std::dynamic_pointer_cast<metadata_action>(find_action(typeid(metadata_action)));
    }

    std::vector<std::shared_ptr<create_branch_action>> metadata_snapshot::branches() const {
        std::vector<std::shared_ptr<create_branch_action>> actions = {};

        for (auto it = find_actions(typeid(create_branch_action)); it != actions_->end(); ++it) {
            actions.push_back(std::dynamic_pointer_cast<create_branch_action>(*it));
        }

        return actions;
    }

    std::shared_ptr<create_branch_action> metadata_snapshot::branch_by_id(const std::string &branch_id) const {
        auto all_branches = branches();

        auto branch = std::ranges::find_if(all_branches.begin(), all_branches.end(),
                                           [branch_id](std::shared_ptr<create_branch_action> b) { return b->id == branch_id; });
        if (branch == all_branches.end()) {
            throw std::runtime_error("Branch id '" + branch_id + "' not found");
        }


        return *branch;
    }

    std::optional<std::string> metadata_snapshot::branch_id(const std::string &name) const {
        for (auto &branch: branches()) {
            if (branch->name == name) {
                return branch->id;
            }
        }
        return std::nullopt;
    }
}
