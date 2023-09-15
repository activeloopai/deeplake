#ifndef DEEPLAKE_REMOVE_FILE_ACTION_HPP
#define DEEPLAKE_REMOVE_FILE_ACTION_HPP

#include <string>
#include "nlohmann/json.hpp"
#include "action.hpp"

namespace deeplake {
    class remove_file_action : public action {
    public:
        remove_file_action(std::string path, long size, long deletion_timestamp, bool data_change);

        remove_file_action(const nlohmann::json &j);

        std::string path();

        long size();

        long deletion_timestamp();

        virtual void to_json(nlohmann::json &json) override;

        arrow::Status append_to(const std::shared_ptr<arrow::StructBuilder> &builder) override;


    private:
        std::string path_;
        long deletion_time_;
        bool data_change_;
        long size_;
    };
}


#endif //DEEPLAKE_REMOVE_FILE_ACTION_HPP
