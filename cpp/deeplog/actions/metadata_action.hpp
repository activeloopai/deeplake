#ifndef DEEPLAKE_METADATA_ACTION_HPP
#define DEEPLAKE_METADATA_ACTION_HPP

#include "action.hpp"
#include "uuid.h"
#include <random>
#include "nlohmann/json.hpp"
#include <arrow/api.h>

namespace deeplake {

    class metadata_action : public action {
    public:
        metadata_action(std::string id, std::optional<std::string> name, std::optional<std::string> description,
                        long created_time);

        metadata_action(const nlohmann::json &j);

        metadata_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        std::string id() const;

        std::optional<std::string> name() const;

        std::optional<std::string> description() const;

        long created_time() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();

    private:
        std::string id_;
        std::optional<std::string> name_;
        std::optional<std::string> description_;
        long created_time_;
    };

    std::string generate_uuid();

    long current_timestamp();
}
#endif //DEEPLAKE_METADATA_ACTION_HPP