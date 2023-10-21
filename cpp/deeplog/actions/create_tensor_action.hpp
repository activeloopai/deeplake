#pragma once

#include "action.hpp"
#include "tensor_link.hpp"

namespace deeplog {
    class create_tensor_action : public action {

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
        std::map<std::string, tensor_link> links;
        std::optional<long> max_chunk_size;
        std::vector<unsigned long> min_shape;
        std::vector<unsigned long> max_shape;
        std::optional<long> tiling_threshold;
        std::optional<std::string> typestr;
        bool verify;
        std::string version;

    public:
        static std::shared_ptr<arrow::StructType> arrow_type;

        create_tensor_action(std::string id,
                             std::string name,
                             std::optional<std::string> dtype,
                             std::string htype,
                             const long &length,
                             const bool &is_link,
                             const bool &is_sequence,
                             const bool &hidden,
                             const std::optional<std::string> &chunk_compression,
                             const std::optional<std::string> &sample_compression,
                             const std::map<std::string, tensor_link> &links,
                             const std::optional<long> &max_chunk_size,
                             const std::vector<unsigned long> &min_shape,
                             const std::vector<unsigned long> &max_shape,
                             const std::optional<long> &tiling_threshold,
                             const std::optional<std::string> &typestr,
                             const bool &verify,
                             std::string version
        );

        explicit create_tensor_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        nlohmann::json to_json() override;

        std::string action_name() override;

        std::shared_ptr<arrow::StructType> action_type() override;
    };
}
