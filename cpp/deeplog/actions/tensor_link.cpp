#include "tensor_link.hpp"

namespace deeplog {
    std::shared_ptr<arrow::StructType> tensor_link::arrow_type = std::dynamic_pointer_cast<arrow::StructType>(
            arrow::struct_({
                                   arrow::field("extend", arrow::utf8(), true),
                                   arrow::field("flatten_sequence", arrow::boolean(), true),
                                   arrow::field("update", arrow::utf8(), true),

                           }));

    tensor_link::tensor_link(const std::string &extend, const std::optional<bool> &flatten_sequence, const std::string &update)
            : extend(extend), flatten_sequence(flatten_sequence), update(update) {}

    tensor_link::tensor_link(const std::shared_ptr<arrow::StructScalar> &value) {
        extend = from_struct<std::string>("extend", value).value();
        flatten_sequence = from_struct<bool>("flatten_sequence", value);
        update = from_struct<std::string>("update", value).value();
    }

    nlohmann::json tensor_link::to_json() const {
        nlohmann::json json;
        json["extend"] = extend;
        json["flatten_sequence"] = to_json_value(flatten_sequence);
        json["update"] = update;
        return json;
    }
}