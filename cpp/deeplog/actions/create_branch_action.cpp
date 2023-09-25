#include "create_branch_action.hpp"

namespace deeplog {
    std::shared_ptr<arrow::DataType> create_branch_action::arrow_struct =  arrow::struct_({
                                                                            arrow::field("id", arrow::utf8()),
                                                                            arrow::field("name", arrow::utf8()),
                                                                            arrow::field("fromId", arrow::utf8()),
                                                                            arrow::field("fromVersion", arrow::int64()),
                                                                    });


    create_branch_action::create_branch_action(std::string id,
                                               std::string name,
                                               std::string from_id,
                                               long from_version) :
            id(id), name(name), from_id(from_id), from_version(from_version) {}

    create_branch_action::create_branch_action(const std::shared_ptr<arrow::StructScalar> &value) {
        id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("id").ValueOrDie())->view();
        name = reinterpret_pointer_cast<arrow::StringScalar>(value->field("name").ValueOrDie())->view();
        from_id = reinterpret_pointer_cast<arrow::StringScalar>(value->field("fromId").ValueOrDie())->view();
        from_version = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("fromVersion").ValueOrDie())->value;
    }

    create_branch_action::create_branch_action(const nlohmann::json &j) {
        const auto &base = j.at("branch");
        base.at("id").get_to(id);
        base.at("name").get_to(name);
        base.at("fromId").get_to(from_id);
        base.at("fromVersion").get_to(from_version);
    }

    void create_branch_action::to_json(nlohmann::json &j) {
        j["branch"]["id"] = id;
        j["branch"]["name"] = name;
        j["branch"]["fromId"] = from_id;
        j["branch"]["fromVersion"] = from_version;
    }

    arrow::Status create_branch_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{id}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::StringScalar{name}));
        ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::StringScalar{from_id}));
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::Int64Scalar{from_version}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }
}