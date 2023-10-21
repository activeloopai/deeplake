#include <arrow/type_fwd.h>
#include <arrow/api.h>
#include "protocol_action.hpp"

namespace deeplog {

    std::shared_ptr<arrow::StructType> protocol_action::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("minReaderVersion", arrow::int32(), true),
                                   arrow::field("minWriterVersion", arrow::int32(), true),
                           }));

    deeplog::protocol_action::protocol_action(const int &min_reader_version, const int &min_writer_version)
            : min_reader_version(min_reader_version), min_writer_version(min_writer_version) {}

    protocol_action::protocol_action(const std::shared_ptr<arrow::StructScalar> &value) {
        min_reader_version = from_struct<int>("minReaderVersion", value).value();
        min_writer_version = from_struct<int>("minWriterVersion", value).value();
    }

    std::string protocol_action::action_name() {
        return "protocol";
    }

    std::shared_ptr<arrow::StructType> protocol_action::action_type() {
        return arrow_type;
    }

    nlohmann::json deeplog::protocol_action::to_json() {
        nlohmann::json json;

        json["minReaderVersion"] = min_reader_version;
        json["minWriterVersion"] = min_writer_version;

        return json;
    }

    bool protocol_action::replaces(std::shared_ptr<action> action) {
        return action->action_name() == action_name();
    }

    std::shared_ptr<action> protocol_action::replace(std::shared_ptr<action> action) {
        return shared_from_this();
    }
}