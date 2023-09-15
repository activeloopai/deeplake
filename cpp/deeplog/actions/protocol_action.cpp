#include <arrow/type_fwd.h>
#include <arrow/api.h>
#include "protocol_action.hpp"

namespace deeplake {
    deeplake::protocol_action::protocol_action(int min_reader_version, int min_writer_version)
            : min_reader_version_(min_reader_version), min_writer_version_(min_writer_version) {}

    protocol_action::protocol_action(const nlohmann::json &j) {
        j.at("protocol").at("minReaderVersion").get_to(min_reader_version_);
        j.at("protocol").at("minWriterVersion").get_to(min_writer_version_);
    }

    protocol_action::protocol_action(const std::shared_ptr<arrow::StructScalar> &value) {
        min_reader_version_ = reinterpret_pointer_cast<arrow::Int32Scalar>(value->field("minReaderVersion").ValueOrDie())->value;
        min_writer_version_ = reinterpret_pointer_cast<arrow::Int32Scalar>(value->field("minWriterVersion").ValueOrDie())->value;
    }

    int protocol_action::min_reader_version() const {
        return min_reader_version_;
    }

    int protocol_action::min_writer_version() const {
        return min_writer_version_;
    }

    void deeplake::protocol_action::to_json(nlohmann::json &j) {
        j["protocol"]["minReaderVersion"] = min_reader_version_;
        j["protocol"]["minWriterVersion"] = min_writer_version_;
    }

    arrow::Status protocol_action::append_to(const std::shared_ptr<arrow::StructBuilder> &builder) {
        ARROW_RETURN_NOT_OK(builder->field_builder(0)->AppendScalar(arrow::Int32Scalar{min_reader_version_}));
        ARROW_RETURN_NOT_OK(builder->field_builder(1)->AppendScalar(arrow::Int32Scalar{min_writer_version_}));

        ARROW_RETURN_NOT_OK(builder->Append());
        return arrow::Status::OK();
    }


    std::shared_ptr<arrow::StructBuilder> deeplake::protocol_action::arrow_array() {
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