#include "deeplog.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <set>
#include <fstream>
#include <nlohmann/json.hpp>
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"
#include "actions/create_tensor_action.hpp"
#include "arrow/io/file.h"
#include "arrow/dataset/api.h"
#include "parquet/stream_writer.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/reader.h"
#include "arrow/util/type_fwd.h"
#include "arrow/table.h"
#include "arrow/array.h"
#include "arrow/builder.h"
#include "arrow/api.h"
#include "arrow/json/api.h"
#include "last_checkpoint.hpp"

namespace deeplog {

    const std::shared_ptr<arrow::Schema> deeplog::arrow_schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
            arrow::field("protocol", protocol_action::arrow_type),
            arrow::field("metadata", metadata_action::arrow_type),
            arrow::field("add", add_file_action::arrow_type),
            arrow::field("branch", create_branch_action::arrow_type),
            arrow::field("version", arrow::uint64()),
    });

    deeplog::deeplog(std::string path) : path(path) {};

    std::shared_ptr<deeplog> deeplog::create(const std::string &path) {
        if (std::filesystem::exists(path)) {
            throw std::runtime_error("'" + path + "' already exists");
        }

        std::filesystem::create_directories(path);
        std::filesystem::create_directories(path + "/_deeplake_log");

        auto log = open(path);
        std::vector<action *> actions;

        auto protocol = std::make_shared<protocol_action>(protocol_action(4, 4));
        auto metadata = std::make_shared<metadata_action>(metadata_action(generate_uuid(), std::nullopt, std::nullopt, current_timestamp()));

        auto branch = std::make_shared<create_branch_action>(create_branch_action(MAIN_BRANCH_ID, "main", MAIN_BRANCH_ID, -1));

        log->commit(MAIN_BRANCH_ID, -1, {protocol, metadata, branch});

        return log;

    }

    std::shared_ptr<deeplog> deeplog::open(const std::string &path) {
        return std::make_shared<deeplog>(deeplog(path));
    }

    int deeplog::log_format() const {
        return 4;
    }

    std::string zero_pad(const long &version) {
        std::ostringstream ss;
        ss << std::setw(20) << std::setfill('0') << (version);
        return ss.str();
    }

    long deeplog::version(const std::string &branch_id) const {
        return get<1>(get_actions(branch_id, std::nullopt));
    }

    void deeplog::commit(const std::string &branch_id,
                         const long &base_version,
                         const std::vector<std::shared_ptr<action>> &actions) {

        auto log_dir = path + "/_deeplake_log/" + branch_id + "/";

        std::filesystem::create_directories(log_dir);

        auto operationFilePath = log_dir + zero_pad(base_version + 1) + ".json";

        std::fstream file(operationFilePath, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + operationFilePath);
        }

        for (auto action: actions) {
            nlohmann::json json;
            json[action->action_name()] = action->to_json();
            file << json;
        }

        file.close();
    }

    arrow::Result<std::shared_ptr<arrow::Table>> deeplog::action_data(const std::string &branch_id,
                                                                          const long &from,
                                                                          const std::optional<long> &to) const {
        long highest_version = -1;
        std::vector<std::shared_ptr<arrow::Table>> all_tables = {};

        const std::filesystem::path dir_path = {path + "/_deeplake_log/" + branch_id};

        std::filesystem::path last_checkpoint_path = {dir_path.string() + "/_last_checkpoint.json"};
        if (std::filesystem::exists(last_checkpoint_path)) {
            auto last_checkpoint_stream = std::ifstream(last_checkpoint_path);
            nlohmann::json last_checkpoint_json = nlohmann::json::parse(last_checkpoint_stream);
            auto checkpoint = last_checkpoint(last_checkpoint_json);

            const arrow::Result<std::shared_ptr<arrow::Table>> &result = read_checkpoint(dir_path.string(), checkpoint.version);
            if (!result.ok()) {
                std::cerr << "Checkpoint read failed: " << result.status() << std::endl;
                return result.status();
            }
            all_tables.push_back(result.ValueOrDie());
            highest_version = checkpoint.version;
        }


        std::optional<long> next_from = from;

//        auto branch_obj = branch_by_id(branch_id).data;
//        all_tables.push_back(action_data(branch_id, from, branch_obj->from_version).ValueOrDie());

//        next_from = branch_obj->from_version + 1;

        std::set < std::filesystem::path > sorted_paths = {};

        auto abs = std::filesystem::absolute(dir_path);
        if (std::filesystem::exists(dir_path)) {
            for (const auto &entry: std::filesystem::directory_iterator(dir_path)) {
                if (std::filesystem::is_regular_file(entry.path()) && entry.path().extension() == ".json" && !entry.path().filename().string().starts_with("_")) {
                    auto found_version = file_version(entry.path());
                    if (to.has_value() && found_version > to) {
                        continue;
                    }

                    if (highest_version < found_version) {
                        highest_version = found_version;
                    }

                    if (!next_from.has_value() || found_version >= next_from) {
                        sorted_paths.insert(entry.path());
                    }
                }
            }
        }

        for (const auto &json_path: sorted_paths) {
            auto file = arrow::io::ReadableFile::Open(json_path);
            auto read_options = arrow::json::ReadOptions::Defaults();
            auto parse_options = arrow::json::ParseOptions::Defaults();
            parse_options.explicit_schema = arrow_schema;

            auto status = arrow::json::TableReader::Make(arrow::default_memory_pool(), file.ValueOrDie(), read_options, parse_options);
            if (!status.ok()) {
                std::cerr << "JSON reader creation failed: " << status.status() << std::endl;
                continue;
            }
            auto json_reader = status.ValueOrDie();


            // Read the JSON data into an Arrow Table
            std::shared_ptr<arrow::Table> arrow_table;
            auto reader_status = json_reader->Read();
            if (!reader_status.ok()) {
                throw std::runtime_error("JSON read failed: " + reader_status.status().message());
            }
            arrow_table = reader_status.ValueOrDie();

            all_tables.push_back(arrow_table);
        }

        std::vector<std::shared_ptr<arrow::Array>> version_row;
        for (const auto &field: arrow_schema->fields()) {
            if (field->name() == "version") {
                version_row.push_back(arrow::MakeArrayFromScalar(arrow::UInt64Scalar(highest_version), 1).ValueOrDie());
            } else {
                version_row.push_back(arrow::MakeArrayOfNull(field->type(), 1).ValueOrDie());
            }
        }
        all_tables.push_back(arrow::Table::Make(arrow_schema, version_row));

        return arrow::ConcatenateTables(all_tables).ValueOrDie();
    }

    std::tuple<std::shared_ptr<std::vector<std::shared_ptr<action>>>, long> deeplog::get_actions(const std::string &branch_id,
                                                                             const std::optional<long> &to) const {
        std::vector<std::shared_ptr<action>> return_actions = {};

        auto all_operations = action_data(branch_id, 0, to).ValueOrDie();

        unsigned long version = -1;
        for (long row_id = 0; row_id < all_operations->num_rows(); ++row_id) {
            auto field_id = 0;
            for (const auto &field: all_operations->fields()) {
                auto scalar = all_operations->column(field_id)->GetScalar(row_id).ValueOrDie();
                if (scalar->is_valid) {
                    if (field->name() == "version") {
                        version = std::dynamic_pointer_cast<arrow::UInt64Scalar>(scalar)->value;
                    } else {
                        std::shared_ptr<::deeplog::action> action;
                        auto struct_scalar = std::dynamic_pointer_cast<arrow::StructScalar>(scalar);
                        if (field->name() == "protocol") {
                            action = std::make_shared<::deeplog::protocol_action>(::deeplog::protocol_action(struct_scalar));
                        } else if (field->name() == "metadata") {
                            action = std::make_shared<::deeplog::metadata_action>(::deeplog::metadata_action(struct_scalar));
                        } else if (field->name() == "branch") {
                            action = std::make_shared<::deeplog::create_branch_action>(::deeplog::create_branch_action(struct_scalar));
                        } else if (field->name() == "add") {
                            action = std::make_shared<::deeplog::add_file_action>(::deeplog::add_file_action(struct_scalar));
                        } else {
                            throw std::runtime_error("Unknown action type: " + field->name());
                        }

                        auto replace_action = std::dynamic_pointer_cast<::deeplog::replace_action>(action);
                        if (replace_action == nullptr) {
                            return_actions.push_back(action);
                        } else {
                            auto matches = std::find_if(return_actions.begin(), return_actions.end(), [replace_action](std::shared_ptr<::deeplog::action> a) {
                                return replace_action->replaces(a);
                            });

                            if (matches == return_actions.end()) {
                                return_actions.push_back(action);
                            } else {
                                auto index = std::distance(return_actions.begin(), matches);
                                auto replacement = replace_action->replace(*matches);
                                if (replacement == nullptr) {
                                    return_actions.erase(return_actions.begin() + index);
                                } else {
                                    return_actions.at(index) = replacement;
                                }
                            }
                        }
                    }
                }
                ++field_id;
            }
        }

        return std::make_tuple(std::make_shared<std::vector<std::shared_ptr<action>>>(return_actions), version);
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

        auto checkpoint_path = path + "/_deeplake_log/_last_checkpoint.json";
        std::fstream file(checkpoint_path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Error opening file: " + checkpoint_path);
        }

        file << checkpoint_json;
        file.close();
    }

    arrow::Result<std::shared_ptr<arrow::Table>> deeplog::read_checkpoint(const std::string &dir_path, const long &version) const {
        arrow::MemoryPool *pool = arrow::default_memory_pool();
        std::shared_ptr<arrow::io::RandomAccessFile> input;
        ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(dir_path + "/" + zero_pad(version) + ".checkpoint.parquet"));

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

        std::shared_ptr<arrow::Table> table;
        ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

        return arrow::Result<std::shared_ptr<arrow::Table>>(table);
    }

    arrow::Status deeplog::write_checkpoint(const std::string &branch_id, const long &version) {
        auto checkpoint_table = action_data(branch_id, 0, version).ValueOrDie();

        std::shared_ptr<parquet::WriterProperties> props = parquet::WriterProperties::Builder().compression(arrow::Compression::SNAPPY)->build();
        std::shared_ptr<parquet::ArrowWriterProperties> arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();

        std::shared_ptr<arrow::io::FileOutputStream> outfile;
        ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(path + "/_deeplake_log/" + zero_pad(version) + ".checkpoint.parquet"));
//
        ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*checkpoint_table, arrow::default_memory_pool(), outfile, 3, props, arrow_props));

        return arrow::Status::OK();
    }

} // deeplake