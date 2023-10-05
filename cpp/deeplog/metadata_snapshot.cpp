#include "metadata_snapshot.hpp"
#include <spdlog/spdlog.h>

namespace deeplog {
    metadata_snapshot::metadata_snapshot(const std::shared_ptr<::deeplog::deeplog> &deeplog) : base_snapshot(META_BRANCH_ID, std::nullopt, deeplog) {
        spdlog::debug("Metadata snapshot created for version {} with {} actions", version, actions_->size());
    }

    metadata_snapshot::metadata_snapshot(const unsigned long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) : base_snapshot(META_BRANCH_ID, version, deeplog) {
        spdlog::debug("Metadata snapshot created for version {} with {} actions", version, actions_->size());
    }

    std::shared_ptr<protocol_action> metadata_snapshot::protocol() const {
        return find_action<protocol_action>();
    }

    std::shared_ptr<metadata_action> metadata_snapshot::metadata() const {
        return find_action<metadata_action>();
    }

    std::vector<std::shared_ptr<create_branch_action>> metadata_snapshot::branches() const {
        return find_actions<create_branch_action>();
    }

    std::shared_ptr<create_branch_action> metadata_snapshot::find_branch(const std::string &address) const {
        auto all_branches = branches();

        std::input_iterator auto branch = std::ranges::find_if(all_branches.begin(), all_branches.end(),
                                           [address](std::shared_ptr<create_branch_action> b) { return b->id == address || b->name == address; });
        if (branch == all_branches.end()) {
            throw std::runtime_error("Branch '" + address + "' not found");
        }


        return *branch;
    }
}
