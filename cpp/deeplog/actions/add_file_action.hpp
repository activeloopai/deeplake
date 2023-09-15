#ifndef DEEPLAKE_ADD_FILE_ACTION_HPP
#define DEEPLAKE_ADD_FILE_ACTION_HPP

#include "action.hpp"

namespace deeplake {
    class add_file_action : public action {

    public:
        add_file_action(std::string path, long size, long modification_time, bool data_change);

        add_file_action(const nlohmann::json &j);

        add_file_action(const std::shared_ptr<arrow::StructScalar> &struct_scalar);

        std::string path() const;

        long size() const;

        long modification_time() const;

        void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;

        static std::shared_ptr<arrow::StructBuilder> arrow_array();


    private:
        std::string path_;
        long size_;
        long modification_time_;
        bool data_change_;
    };
}


#endif //DEEPLAKE_ADD_FILE_ACTION_HPP
