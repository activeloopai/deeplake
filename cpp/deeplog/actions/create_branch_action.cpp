#include "create_branch_action.hpp"

namespace deeplog {
    create_branch_action::create_branch_action(std::string id,
                                               std::string name,
                                               std::string from_branch,
                                               long from_version) :
            id(id), name(name), from_id(from_branch), from_version(from_version) {}

    create_branch_action::create_branch_action(const nlohmann::json &j) {
        const auto &base = j.at("createBranch");
        base.at("id").get_to(id);
        base.at("name").get_to(name);
        base.at("fromId").get_to(from_id);
        base.at("fromVersion").get_to(from_version);
    }

    void create_branch_action::to_json(nlohmann::json &j) {
        j["createBranch"]["id"] = id;
        j["createBranch"]["name"] = name;
        j["createBranch"]["fromId"] = from_id;
        j["createBranch"]["fromVersion"] = from_version;
    }

    arrow::Status create_branch_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        return arrow::Status::OK();
    }
}