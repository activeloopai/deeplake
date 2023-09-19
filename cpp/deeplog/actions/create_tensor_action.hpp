#pragma once

#include "action.hpp"

namespace deeplog {
    class create_tensor_action : public action {

    public:
        create_tensor_action(std::string id,
                             std::string name,
                             std::string dtype,
                             std::string htype,
                             long length,
                             bool link,
                             bool sequence,
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

        std::string id() const;

        std::string name() const;

        std::string dtype() const;

        std::string htype() const;

        long length() const;

        bool link() const;

        bool sequence() const;

        bool hidden() const;

        std::optional<std::string> chunk_compression() const;

        std::optional<std::string> sample_compression() const;

        std::map<std::string, std::map<std::string, std::string>> links() const;

        std::optional<long> max_chunk_size() const;

        std::vector<long> min_shape() const;

        std::vector<long> max_shape() const;

        std::optional<long> tiling_threshold() const;

        std::optional<std::string> typestr() const;

        bool verify() const;

        std::string version() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();


    private:
        std::string id_;
        std::string name_;
        std::string dtype_;
        std::string htype_;
        long length_;
        bool link_;
        bool sequence_;
        bool hidden_;
        std::optional<std::string> chunk_compression_;
        std::optional<std::string> sample_compression_;
        std::map<std::string, std::map<std::string, std::string>> links_;
        std::optional<long> max_chunk_size_;
        std::vector<long> min_shape_;
        std::vector<long> max_shape_;
        std::optional<long> tiling_threshold_;
        std::optional<std::string> typestr_;
        bool verify_;
        std::string version_;
    };
}
