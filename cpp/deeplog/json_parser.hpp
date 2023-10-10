#pragma once

#include <arrow/api.h>
#include <arrow/io/api.h>

namespace deeplog {
    class json_parser {
    public:
        static arrow::Status parse(const std::shared_ptr<arrow::io::InputStream> &json, const std::shared_ptr<arrow::RecordBatchBuilder> &batch_builder);

    private:
        static arrow::Result<std::shared_ptr<arrow::Scalar>> convert(const std::shared_ptr<arrow::Scalar> &current_value, const std::shared_ptr<arrow::DataType> &wanted_type);

    };
}
