#include "deeplog.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <fstream>
#include "nlohmann/json.hpp"
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"
#include "arrow/io/file.h"
#include "parquet/stream_writer.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/reader.h"
#include "arrow/util/type_fwd.h"
#include "arrow/table.h"
#include "arrow/array.h"
#include "arrow/builder.h"
#include "arrow/api.h"
#include "last_checkpoint.hpp"

namespace deeplake {

    deeplog::deeplog(std::string path) : path_(path) {};

    std::shared_ptr<deeplake::deeplog> deeplog::create(const std::string &path) {
        if (std::filesystem::exists(path)) {
            throw std::runtime_error("'" + path + "' already exists");
        }

        std::filesystem::create_directories(path);
        std::filesystem::create_directories(path + "/_deeplake_log");

        auto log = open(path);
        std::vector<action *> actions;

        auto protocol = deeplake::protocol_action(4, 4);
        auto metadata = deeplake::metadata_action(generate_uuid(), std::nullopt, std::nullopt, current_timestamp());

        auto branch = deeplake::create_branch_action(MAIN_BRANCH_ID, "main", MAIN_BRANCH_ID, -1);

        log->commit(MAIN_BRANCH_ID, -1, {&protocol, &metadata, &branch});

        return log;

    }

    std::shared_ptr<deeplake::deeplog> deeplog::open(const std::string &path) {
        return std::make_shared<deeplake::deeplog>(deeplog(path));
    }

    std::string deeplog::path() { return path_; }

    long deeplog::version() const {
        return version(MAIN_BRANCH_ID);
    }

    std::string zero_pad(const long &version) {
        std::ostringstream ss;
        ss << std::setw(20) << std::setfill('0') << (version);
        return ss.str();
    }

    long deeplog::version(const std::string &branch_id) const {
        return list_actions(branch_id, 0, std::nullopt).version;
    }

    deeplog_state<std::shared_ptr<deeplake::protocol_action>> deeplog::protocol() const {
        auto actions = list_actions(MAIN_BRANCH_ID, 0, std::nullopt);

        std::shared_ptr<protocol_action> protocol;

        for (auto action: actions.data) {
            auto casted = std::dynamic_pointer_cast<protocol_action>(action);
            if (casted != nullptr) {
                protocol = casted;
            }
        }

        return {protocol, actions.version};
    }

    deeplog_state<std::shared_ptr<deeplake::metadata_action>> deeplog::metadata() const {
        auto actions = list_actions(MAIN_BRANCH_ID, 0, std::nullopt);

        std::shared_ptr<metadata_action> metadata;

        for (auto action: actions.data) {
            auto casted = std::dynamic_pointer_cast<metadata_action>(action);
            if (casted != nullptr) {
                metadata = casted;
            }
        }

        return {metadata, actions.version};
    }

    deeplog_state<std::vector<std::shared_ptr<deeplake::add_file_action>>> deeplog::data_files(const std::string &branch_id, const std::optional<long> &version) {
        auto actions = list_actions(MAIN_BRANCH_ID, 0, std::nullopt);

        std::vector<std::shared_ptr<add_file_action>> branches = {};

        for (const auto action: actions.data) {
            auto casted = std::dynamic_pointer_cast<add_file_action>(action);
            if (casted != nullptr) {
                branches.push_back(casted);
            }
        }

        return {std::vector<std::shared_ptr<add_file_action>>(branches), actions.version};
    }

    deeplog_state<std::vector<std::shared_ptr<deeplake::create_branch_action>>> deeplog::branches() const {
        auto actions = list_actions(MAIN_BRANCH_ID, 0, std::nullopt);

        std::vector<std::shared_ptr<create_branch_action>> branches = {};

        for (const auto action: actions.data) {
            auto casted = std::dynamic_pointer_cast<create_branch_action>(action);
            if (casted != nullptr) {
                branches.push_back(casted);
            }
        }

        return {std::vector<std::shared_ptr<create_branch_action>>(branches), actions.version};
    }


    void deeplog::commit(const std::string &branch_id,
                         const long &base_version,
                         const std::vector<deeplake::action *> &actions) {
        nlohmann::json commit_json;

        for (auto action: actions) {
            nlohmann::json action_json = nlohmann::json::object();
            action->to_json(action_json);
            commit_json.push_back(action_json);
        }

        auto log_dir = path_ + "/_deeplake_log/" + branch_id + "/";

        std::filesystem::create_directories(log_dir);

        auto operationFilePath = log_dir + zero_pad(base_version + 1) + ".json";

        std::fstream file(operationFilePath, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + operationFilePath);
        }

        file << commit_json;
        file.close();
    }

    deeplog_state<std::shared_ptr<deeplake::create_branch_action>> deeplog::branch_by_id(const std::string &branch_id) const {
        auto all_branches = this->branches();
        auto data = all_branches.data;

        auto branch = std::ranges::find_if(data,
                                           [branch_id](std::shared_ptr<deeplake::create_branch_action> b) { return b->id() == branch_id; });
        if (branch == data.end()) {
            throw std::runtime_error("Branch id '" + branch_id + "' not found");
        }

        return {*branch, all_branches.version};
    }

    deeplog_state<std::vector<std::shared_ptr<action>>> deeplog::list_actions(const std::string &branch_id,
                                                                              const long &from,
                                                                              const std::optional<long> &to) const {
        long higheset_version = -1;
        std::vector<std::shared_ptr<action>> return_actions = {};

        const std::filesystem::path dir_path = {path_ + "/_deeplake_log/" + branch_id};

        std::filesystem::path last_checkpoint_path = {dir_path.string() + "/_last_checkpoint.json"};
        if (std::filesystem::exists(last_checkpoint_path)) {
            auto last_checkpoint_stream = std::ifstream(last_checkpoint_path);
            nlohmann::json last_checkpoint_json = nlohmann::json::parse(last_checkpoint_stream);
            auto checkpoint = last_checkpoint(last_checkpoint_json);

            auto status = read_checkpoint(dir_path.string(), checkpoint.version, return_actions);
            higheset_version = checkpoint.version;
        }


        std::optional<long> next_from = from;

        if (branch_id != MAIN_BRANCH_ID) {
            auto branch_obj = branch_by_id(branch_id).data;
            for (const auto &action: list_actions(branch_id, from, branch_obj->from_version()).data) {
                return_actions.push_back(action);
            }

            next_from = branch_obj->from_version() + 1;
        }

        std::set < std::filesystem::path > sorted_paths = {};

        if (std::filesystem::exists(dir_path)) {
            for (const auto &entry: std::filesystem::directory_iterator(dir_path)) {
                if (std::filesystem::is_regular_file(entry.path()) && entry.path().extension() == ".json" && !entry.path().filename().string().starts_with("_")) {
                    auto found_version = file_version(entry.path());
                    if (higheset_version < found_version) {
                        higheset_version = found_version;
                    }
                    if (to.has_value() && found_version > to) {
                        continue;
                    }
                    if (!next_from.has_value() || found_version >= next_from) {
                        sorted_paths.insert(entry.path());
                    }
                }
            }
        }

        for (const auto &path: sorted_paths) {
            std::ifstream ifs(path);

            nlohmann::json jsonArray = nlohmann::json::parse(ifs);
            for (auto &element: jsonArray) {
                if (element.contains("add")) {
                    return_actions.push_back(std::make_shared<add_file_action>(deeplake::add_file_action(element)));
                } else if (element.contains("createBranch")) {
                    return_actions.push_back(std::make_shared<create_branch_action>(deeplake::create_branch_action(element)));
                } else if (element.contains("protocol")) {
                    return_actions.push_back(std::make_shared<protocol_action>(deeplake::protocol_action(element)));
                } else if (element.contains("metadata")) {
                    return_actions.push_back(std::make_shared<metadata_action>(deeplake::metadata_action(element)));
                }
            }
        }

        return deeplog_state(return_actions, higheset_version);
    }

    long deeplog::file_version(const std::filesystem::path &path) const {
        auto formatted_version = path.filename().string()
                .substr(0, path.filename().string().length() - 5);
        return std::stol(formatted_version);
    }

    void deeplog::checkpoint(const std::string &branch_id) {
        long version_to_checkpoint = version(branch_id);

        auto status = write_checkpoint(branch_id, version_to_checkpoint);

        if (!status.ok()) {
            std::cout << status.message() << std::endl;
            return;
        }
        nlohmann::json checkpoint_json = last_checkpoint(version_to_checkpoint, 3013);

        auto checkpoint_path = path_ + "/_deeplake_log/_last_checkpoint.json";
        std::fstream file(checkpoint_path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + checkpoint_path);
        }

        file << checkpoint_json;
        file.close();
    }

    arrow::Status deeplog::read_checkpoint(const std::string &dir_path, const long &version, std::vector<std::shared_ptr<deeplake::action>> &actions) const {


        auto reader_properties = parquet::ReaderProperties(arrow::default_memory_pool());
        reader_properties.set_buffer_size(4096 * 4);
        reader_properties.enable_buffered_stream();

        parquet::arrow::FileReaderBuilder reader_builder;
        ARROW_RETURN_NOT_OK(reader_builder.OpenFile(dir_path + "/" + zero_pad(version) + ".checkpoint.parquet", false, reader_properties));
        reader_builder.memory_pool(arrow::default_memory_pool());

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        ARROW_ASSIGN_OR_RAISE(arrow_reader, reader_builder.Build());

        std::shared_ptr<arrow::RecordBatchReader> rb_reader;
        ARROW_RETURN_NOT_OK(arrow_reader->GetRecordBatchReader(&rb_reader));

        for (arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_batch: *rb_reader) {
            std::shared_ptr<arrow::RecordBatch> batch;
            ARROW_ASSIGN_OR_RAISE(batch, maybe_batch);

            for (auto i = 0; i < batch->num_rows(); ++i) {
                if (!batch->GetColumnByName("protocol")->data()->IsNull(i)) {
                    std::shared_ptr<arrow::Scalar> val;
                    ARROW_ASSIGN_OR_RAISE(val, batch->GetColumnByName("protocol")->GetScalar(i));

                    actions.push_back(std::make_shared<deeplake::protocol_action>(protocol_action(reinterpret_pointer_cast<arrow::StructScalar>(val))));
                }

                if (!batch->GetColumnByName("metadata")->data()->IsNull(i)) {
                    std::shared_ptr<arrow::Scalar> val;
                    ARROW_ASSIGN_OR_RAISE(val, batch->GetColumnByName("metadata")->GetScalar(i));

                    actions.push_back(std::make_shared<deeplake::metadata_action>(metadata_action(reinterpret_pointer_cast<arrow::StructScalar>(val))));
                }

                if (!batch->GetColumnByName("add")->data()->IsNull(i)) {
                    std::shared_ptr<arrow::Scalar> val;
                    ARROW_ASSIGN_OR_RAISE(val, batch->GetColumnByName("add")->GetScalar(i));

                    actions.push_back(std::make_shared<deeplake::add_file_action>(add_file_action(reinterpret_pointer_cast<arrow::StructScalar>(val))));
                }
            }
        }

        return arrow::Status::OK();
    }

    arrow::Status deeplog::write_checkpoint(const std::string &branch_id, const long &version) {
        auto protocol_builder = deeplake::protocol_action::arrow_array();
        auto metadata_builder = deeplake::metadata_action::arrow_array();
        auto add_file_builder = deeplake::add_file_action::arrow_array();

        ARROW_RETURN_NOT_OK(protocol().data->append_to(protocol_builder));
        ARROW_RETURN_NOT_OK(metadata_builder->AppendNull());
        ARROW_RETURN_NOT_OK(add_file_builder->AppendNull());

        ARROW_RETURN_NOT_OK(protocol_builder->AppendNull());
        ARROW_RETURN_NOT_OK(metadata().data->append_to(metadata_builder));
        ARROW_RETURN_NOT_OK(add_file_builder->AppendNull());

        for (auto file: data_files(branch_id, version).data) {
            ARROW_RETURN_NOT_OK(protocol_builder->AppendNull());
            ARROW_RETURN_NOT_OK(metadata_builder->AppendNull());
            ARROW_RETURN_NOT_OK(file->append_to(add_file_builder));
        }

        std::shared_ptr<arrow::Array> protocol_array, metadata_array, add_file_array;
        ARROW_RETURN_NOT_OK(protocol_builder->Finish(&protocol_array));
        ARROW_RETURN_NOT_OK(metadata_builder->Finish(&metadata_array));
        ARROW_RETURN_NOT_OK(add_file_builder->Finish(&add_file_array));

        auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
                arrow::field("protocol", protocol_builder->type()),
                arrow::field("metadata", metadata_builder->type()),
                arrow::field("add", add_file_builder->type()),
        });
        const auto checkpoint_table = arrow::Table::Make(schema, {protocol_array, metadata_array, add_file_array});

        std::shared_ptr<parquet::WriterProperties> props = parquet::WriterProperties::Builder().compression(arrow::Compression::SNAPPY)->build();
        std::shared_ptr<parquet::ArrowWriterProperties> arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

        std::shared_ptr<arrow::io::FileOutputStream> outfile;
        ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path_ + "/_deeplake_log/" + zero_pad(version) + ".checkpoint.parquet"));
//
        ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*checkpoint_table, arrow::default_memory_pool(), outfile, 3, props, arrow_props));

        return arrow::Status::OK();
    }

} // deeplake