#ifndef DEEPLAKE_PROTOCOL_ACTION_HPP
#define DEEPLAKE_PROTOCOL_ACTION_HPP

#include "action.hpp"
#include <arrow/api.h>

namespace deeplake {

    class protocol_action : public action {
    public:
        static std::shared_ptr<arrow::StructBuilder> arrow_array();

        protocol_action(int min_reader_version, int min_writer_version);

        protocol_action(const nlohmann::json &j);

        protocol_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        int min_reader_version() const;

        int min_writer_version() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

    private:
        int min_reader_version_;
        int min_writer_version_;
    };

}
#endif //DEEPLAKE_PROTOCOL_ACTION_HPP
