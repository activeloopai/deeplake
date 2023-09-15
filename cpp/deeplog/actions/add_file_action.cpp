#include "add_file_action.hpp"

namespace deeplake {
    add_file_action::add_file_action(std::string path, long size, long modification_time, bool data_change) :
            path_(path), size_(size), modification_time_(modification_time), data_change_(data_change) {}

    add_file_action::add_file_action(const nlohmann::json &j) {
        const auto &base = j.at("add");
        base.at("path").get_to(path_);
        base.at("size").get_to(size_);
        base.at("modificationTime").get_to(modification_time_);
        base.at("dataChange").get_to(data_change_);
    }

    add_file_action::add_file_action(const std::shared_ptr<arrow::StructScalar> &value) {
        path_ = reinterpret_pointer_cast<arrow::StringScalar>(value->field("path").ValueOrDie())->view();
        size_ = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("size").ValueOrDie())->value;
        modification_time_ = reinterpret_pointer_cast<arrow::Int64Scalar>(value->field("modificationTime").ValueOrDie())->value;
        data_change_ = reinterpret_pointer_cast<arrow::BooleanScalar>(value->field("dataChange").ValueOrDie())->value;
    }

    std::string add_file_action::path() const { return path_; }

    long add_file_action::size() const { return size_; }

    long add_file_action::modification_time() const { return modification_time_; }

    void add_file_action::to_json(nlohmann::json &j) {
        j["add"]["path"] = path_;
        j["add"]["size"] = size_;
        j["add"]["modificationTime"] = modification_time_;
        j["add"]["dataChange"] = data_change_;
    }

    std::shared_ptr<arrow::StructBuilder> deeplake::add_file_action::arrow_array() {
        auto protocol_struct = arrow::struct_({
                                                      arrow::field("path", arrow::utf8()),
                                                      arrow::field("size", arrow::uint64()),
                                                      arrow::field("modificationTime", arrow::uint64()),
                                                      arrow::field("dataChange", arrow::boolean()),
                                              });

        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(protocol_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::StringBuilder>(arrow::StringBuilder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::Int64Builder>(arrow::Int64Builder()),
                std::make_shared<arrow::BooleanBuilder>(arrow::BooleanBuilder()),
        })));

    }

    arrow::Status add_file_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::StringScalar{path_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::Int64Scalar{size_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(2)->AppendScalar(arrow::Int64Scalar{modification_time_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(3)->AppendScalar(arrow::BooleanScalar{data_change_}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }

}