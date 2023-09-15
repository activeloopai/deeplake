#ifndef DEEPLAKE_CREATE_BRANCH_ACTION_HPP
#define DEEPLAKE_CREATE_BRANCH_ACTION_HPP

#include "action.hpp"

namespace deeplake {
    class create_branch_action : public action {

    public:
        create_branch_action(std::string id, std::string name, std::string from_branch, long from_version);

        create_branch_action(const nlohmann::json &j);

        std::string id() const;

        std::string name() const;

        std::string from_id() const;

        long from_version() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;


    private:
        std::string id_;
        std::string name_;
        std::string from_branch_id_;
        long from_version_;

    };
}


#endif //DEEPLAKE_CREATE_BRANCH_ACTION_HPP
