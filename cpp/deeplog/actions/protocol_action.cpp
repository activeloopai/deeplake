#include <arrow/type_fwd.h>
#include <arrow/api.h>
#include "protocol_action.hpp"

namespace deeplog {
    deeplog::protocol_action::protocol_action(int min_reader_version, int min_writer_version)
            : min_reader_version(min_reader_version), min_writer_version(min_writer_version) {}

    protocol_action::protocol_action(const nlohmann::json &j) {
        j.at("protocol").at("minReaderVersion").get_to(min_reader_version);
        j.at("protocol").at("minWriterVersion").get_to(min_writer_version);
    }

    protocol_action::protocol_action(const std::shared_ptr<arrow::StructScalar> &value) {
        min_reader_version = reinterpret_pointer_cast<arrow::Int32Scalar>(value->field("minReaderVersion").ValueOrDie())->value;
        min_writer_version = reinterpret_pointer_cast<arrow::Int32Scalar>(value->field("minWriterVersion").ValueOrDie())->value;
    }

    void deeplog::protocol_action::to_json(nlohmann::json &j) {
        j["protocol"]["minReaderVersion"] = min_reader_version;
        j["protocol"]["minWriterVersion"] = min_writer_version;
    }

    arrow::Status protocol_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::Int32Scalar{min_reader_version}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::Int32Scalar{min_writer_version}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }


    std::shared_ptr<arrow::StructBuilder> deeplog::protocol_action::arrow_array() {
        auto protocol_struct = arrow::struct_({
                                                      arrow::field("minReaderVersion", arrow::int32()),
                                                      arrow::field("minWriterVersion", arrow::int32()),
                                              });

        return std::make_shared<arrow::StructBuilder>(std::move(arrow::StructBuilder(protocol_struct, arrow::default_memory_pool(), {
                std::make_shared<arrow::Int32Builder>(arrow::Int32Builder()),
                std::make_shared<arrow::Int32Builder>(arrow::Int32Builder()),
        })));

    }
}