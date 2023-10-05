#include "base_snapshot.hpp"

namespace deeplog {

    base_snapshot::base_snapshot(const std::string &branch_id, const std::optional<long> &version, const std::shared_ptr<::deeplog::deeplog> &deeplog) :
            version(0),
            deeplog(deeplog),
            actions_() {
        std::tie(actions_, this->version) = deeplog->get_actions(branch_id, version);
    }

    template<typename T>
    std::vector<std::shared_ptr<T>> base_snapshot::find_actions() const {
        static_assert(std::is_base_of<action, T>::value, "T must be a subclass of action");

        std::vector<std::shared_ptr<T>> return_actions = {};
        for (auto found : *actions_) {
            auto casted = std::dynamic_pointer_cast<T>(found);
            if (casted != nullptr) {
                return_actions.push_back(casted);
            }
        }

        return return_actions;

    }

    template<>
    std::vector<std::shared_ptr<create_tensor_action>> base_snapshot::find_actions() const {
        std::vector<std::shared_ptr<create_tensor_action>> return_actions = {};
        std::map<std::string, int> tensor_idxs = {};
        for (auto found : *actions_) {
            auto casted = std::dynamic_pointer_cast<create_tensor_action>(found);
            if (casted != nullptr) {
                if (tensor_idxs.find(casted->id) != tensor_idxs.end()) {
                    return_actions.at(tensor_idxs[casted->id]) = casted;
                } else {
                    tensor_idxs[casted->id] = return_actions.size();
                    return_actions.push_back(casted);
                }
            }
        }

        return return_actions;
    }

    template<typename T>
    std::shared_ptr<T> base_snapshot::find_action() const {
        static_assert(std::is_base_of<action, T>::value, "T must be a subclass of action");

        auto actions = find_actions<T>();

        if (actions.empty()) {
            return nullptr;
        }
        return actions.at(0);
    }


    template std::vector<std::shared_ptr<create_branch_action>> base_snapshot::find_actions() const;

    template std::vector<std::shared_ptr<create_tensor_action>> base_snapshot::find_actions() const;

    template std::vector<std::shared_ptr<create_commit_action>> base_snapshot::find_actions() const;

    template std::vector<std::shared_ptr<add_file_action>> base_snapshot::find_actions() const;

    template std::shared_ptr<metadata_action> base_snapshot::find_action() const;

    template std::shared_ptr<protocol_action> base_snapshot::find_action() const;
}