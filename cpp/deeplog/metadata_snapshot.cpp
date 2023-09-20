#include "metadata_snapshot.hpp"

namespace deeplog {
    metadata_snapshot::metadata_snapshot(const std::shared_ptr<deeplog> &deeplog) :
            version(deeplog->version(MAIN_BRANCH_ID)),
            deeplog_(deeplog) {
    }

    metadata_snapshot::metadata_snapshot(const long &version, const std::shared_ptr<deeplog> &deeplog) :
            version(version),
            deeplog_(deeplog) {
    }

    std::shared_ptr<protocol_action> metadata_snapshot::protocol() const {
        return deeplog_->protocol().data;
    }

    std::shared_ptr<metadata_action> metadata_snapshot::metadata() const {
        return deeplog_->metadata().data;
    }

    std::vector<std::shared_ptr<create_branch_action>> metadata_snapshot::branches() const {
        return deeplog_->branches().data;
    }

    std::shared_ptr<create_branch_action> metadata_snapshot::branch_by_id(const std::string &branch_id) const {
        return deeplog_->branch_by_id(branch_id).data;
    }

    std::optional<std::string> metadata_snapshot::branch_id(const std::string &name) const {
        for (auto &branch : branches()) {
            if (branch->name() == name) {
                return branch->id();
            }
        }
        return std::nullopt;
    }
}
