#pragma once

#include "action.hpp"

namespace deeplog {
    class create_tensor_action : public action {

    public:
        static std::shared_ptr<arrow::DataType> arrow_struct;

        create_tensor_action(std::string id,
                             std::string name,
                             std::optional<std::string> dtype,
                             std::string htype,
                             long length,
                             bool is_link,
                             bool is_sequence,
                             bool hidden,
                             std::optional<std::string> chunk_compression,
                             std::optional<std::string> sample_compression,
                             std::map<std::string, std::map<std::string, std::string>> links,
                             std::optional<long> max_chunk_size,
                             std::vector<long> min_shape,
                             std::vector<long> max_shape,
                             std::optional<long> tiling_threshold,
                             std::optional<std::string> typestr,
                             bool verify,
                             std::string version
        );

        create_tensor_action(const nlohmann::json &j);

        create_tensor_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();


    public:
        std::string id;
        std::string name;
        std::optional<std::string> dtype;
        std::string htype;
        long length;
        bool is_link;
        bool is_sequence;
        bool hidden;
        std::optional<std::string> chunk_compression;
        std::optional<std::string> sample_compression;
        std::map<std::string, std::map<std::string, std::string>> links;
        std::optional<long> max_chunk_size;
        std::vector<long> min_shape;
        std::vector<long> max_shape;
        std::optional<long> tiling_threshold;
        std::optional<std::string> typestr;
        bool verify;
        std::string version;
    };
}
