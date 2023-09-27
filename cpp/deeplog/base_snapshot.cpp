#include "base_snapshot.hpp"

namespace deeplog {
    base_snapshot::base_snapshot(const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            version(-1),
            deeplog(deeplog),
            actions_() {
        std::tie(actions_, version) = deeplog->get_actions(MAIN_BRANCH_ID, std::nullopt);
    }

    base_snapshot::base_snapshot(const long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            version(version),
            deeplog(deeplog),
            actions_() {
        std::tie(actions_, this->version) = deeplog->get_actions(MAIN_BRANCH_ID, version);
    }

    std::vector<std::shared_ptr<action>>::iterator base_snapshot::find_actions(const std::type_info &type) const {
        return std::find_if(actions_->begin(), actions_->end(), [&type](const std::shared_ptr<::deeplog::action> &action) {
            return typeid(*action) == type;
        });
    }

    std::shared_ptr<action> base_snapshot::find_action(const std::type_info &type) const {
        auto actions = find_actions(type);

        if (actions == actions_->end()) {
            return nullptr;
        }
        return *actions;
    }
}