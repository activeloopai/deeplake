#include "deeplog.hpp"
#include <algorithm>
#include <set>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include "actions/protocol_action.hpp"
#include "actions/metadata_action.hpp"
#include "actions/create_branch_action.hpp"
#include "actions/create_tensor_action.hpp"
#include <arrow/io/memory.h>
#include <arrow/dataset/api.h>
#include <parquet/stream_writer.h>
#include <parquet/arrow/writer.h>
#include <parquet/arrow/reader.h>
#include <arrow/util/type_fwd.h>
#include <arrow/api.h>
#include <arrow/json/api.h>
#include "last_checkpoint.hpp"
#include "deeplog_v3.hpp"
#include "../storage/local_storage.hpp"

namespace deeplog {

    const std::shared_ptr<arrow::Schema> deeplog::arrow_schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
            arrow::field("protocol", protocol_action::arrow_type),
            arrow::field("metadata", metadata_action::arrow_type),
            arrow::field("add", add_file_action::arrow_type),
            arrow::field("branch", create_branch_action::arrow_type),
            arrow::field("tensor", create_tensor_action::arrow_type),
            arrow::field("version", arrow::uint64()),
    });

    deeplog::deeplog(const std::shared_ptr<storage::storage> &storage) : storage_(storage) {};

    std::shared_ptr<deeplog> deeplog::create(const std::string &path, const int &log_version) {
        return create(std::make_shared<storage::local_storage>(storage::local_storage(path)), log_version);
    }

    std::shared_ptr<deeplog> deeplog::create(const std::shared_ptr<::storage::storage> &storage, const int &log_version) {
        if (log_version < 3) {
            throw std::runtime_error("Log version " + std::to_string(log_version) + " is not supported");
        }
        if (log_version == 3) {
            return std::make_shared<deeplog_v3>(deeplog_v3(storage));
        }

        if (storage->file("/_deeplake_log").exists()) {
            throw std::runtime_error("DeepLake config already exists");
        }

        auto log = std::make_shared<deeplog>(deeplog(storage));
        std::vector<action *> actions;

        auto protocol = std::make_shared<protocol_action>(protocol_action(4, 4));
        auto metadata = std::make_shared<metadata_action>(metadata_action(generate_uuid(), std::nullopt, std::nullopt, current_timestamp()));

        auto branch = std::make_shared<create_branch_action>(create_branch_action(MAIN_BRANCH_ID, "main", MAIN_BRANCH_ID, -1));

        log->commit(MAIN_BRANCH_ID, -1, {protocol, metadata, branch});

        return log;

    }

    std::shared_ptr<deeplog> deeplog::open(const std::string &path) {
        spdlog::debug("Opening log at path: {}", std::filesystem::absolute(path).string());
        return open(std::make_shared<storage::local_storage>(storage::local_storage(path)));
    }

    std::shared_ptr<deeplog> deeplog::open(const std::shared_ptr<storage::storage> &storage) {
        if (!storage->file("/_deeplake_log").exists()) {
            if (storage->file("/dataset_meta.json").exists()) {
                return std::make_shared<deeplog_v3>(deeplog_v3(storage));
            }
            throw std::runtime_error("Cannot determine log format");
        }

        return std::make_shared<deeplog>(deeplog(storage));
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

        auto log_dir = "/_deeplake_log/" + branch_id + "/";

        auto operationFilePath = log_dir + zero_pad(base_version + 1) + ".json";

        spdlog::debug("Committing {} actions to {}", actions.size(), operationFilePath);

        std::stringstream buffer;
        for (auto action: actions) {
            nlohmann::json json;
            json[action->action_name()] = action->to_json();
            buffer << json;
        }

        storage_->set_bytes(operationFilePath, buffer.str());
    }

    arrow::Result<std::shared_ptr<arrow::Table>> deeplog::action_data(const std::string &branch_id,
                                                                      const long &from,
                                                                      const std::optional<long> &to) const {
        spdlog::debug("Reading action data for branch '{}' from {} to {}", branch_id, from, to.value_or(9999999));
        long highest_version = -1;
        std::vector<std::shared_ptr<arrow::Table>> all_tables = {};

        const auto dir_path = "/_deeplake_log/" + branch_id;

        auto last_checkpoint_path = "/_deeplake_log/_last_checkpoint.json";
        auto last_checkpoint_ref = storage_->file(last_checkpoint_path);
        if (last_checkpoint_ref.exists()) {
            auto last_checkpoint_stream = storage_->get_bytes(last_checkpoint_path);
            nlohmann::json last_checkpoint_json = nlohmann::json::parse(last_checkpoint_stream);
            auto checkpoint = last_checkpoint(last_checkpoint_json);

            const arrow::Result<std::shared_ptr<arrow::Table>> &result = read_checkpoint(dir_path, checkpoint.version);
            if (!result.ok()) {
                spdlog::error("Checkpoint read failed: {}", result.status().message());
                return result.status();
            }
            all_tables.push_back(result.ValueOrDie());
            highest_version = checkpoint.version;
        }


        std::optional<long> next_from = from;

//        auto branch_obj = branch_by_id(branch_id).data;
//        all_tables.push_back(action_data(branch_id, from, branch_obj->from_version).ValueOrDie());

//        next_from = branch_obj->from_version + 1;

        std::set<::storage::file_ref> sorted_paths = {};

        if (storage_->file(dir_path).exists()) {
            for (auto file_ref: storage_->list_files(dir_path)) {
                if (file_ref.path.ends_with(".json") && !file_ref.path.ends_with("_last_checkpoint.json")) {
                    auto found_version = file_version(file_ref.path);
                    if (to.has_value() && found_version > to) {
                        continue;
                    }

                    if (highest_version < found_version) {
                        highest_version = found_version;
                    }

                    if (!next_from.has_value() || found_version >= next_from) {
                        sorted_paths.insert(file_ref);
                    }
                }
            }
        }

        for (const auto &json_path: sorted_paths) {
            spdlog::debug("Reading data from {}", json_path.path);
            auto buffer_reader = open_arrow_istream(json_path);

            auto read_options = arrow::json::ReadOptions::Defaults();
            auto parse_options = arrow::json::ParseOptions::Defaults();
            parse_options.explicit_schema = arrow_schema;

            auto status = arrow::json::TableReader::Make(arrow::default_memory_pool(), buffer_reader, read_options, parse_options);
            if (!status.ok()) {
                spdlog::error("JSON reader creation failed: {}", status.status().message());
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

        spdlog::debug("Finished loading data in {} to version {}", branch_id, highest_version);
        all_tables.push_back(arrow::Table::Make(arrow_schema, version_row));

        return arrow::ConcatenateTables(all_tables).ValueOrDie();
    }

    std::tuple<std::shared_ptr<std::vector<std::shared_ptr<action>>>, long> deeplog::get_actions(const std::string &branch_id,
                                                                                                 const std::optional<long> &to) const {
        std::vector<std::shared_ptr<action>> return_actions = {};

        auto all_operations = action_data(branch_id, 0, to).ValueOrDie();

        spdlog::debug("Parsing action data...");

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
                        } else if (field->name() == "tensor") {
                            action = std::make_shared<::deeplog::create_tensor_action>(::deeplog::create_tensor_action(struct_scalar));
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

        spdlog::debug("Loaded {} actions for branch '{}' to version {}", return_actions.size(), branch_id, version);

        return std::make_tuple(std::make_shared<std::vector<std::shared_ptr<action>>>(return_actions), version);
    }

    long deeplog::file_version(const std::string &path) const {
        std::filesystem::path path_obj = path;
        auto formatted_version = path_obj.filename().string()
                .substr(0, path_obj.filename().string().length() - 5);
        return std::stol(formatted_version);
    }

    void deeplog::checkpoint(const std::string &branch_id) {
        long version_to_checkpoint = version(branch_id);

        auto status = write_checkpoint(branch_id, version_to_checkpoint);

        if (!status.ok()) {
            spdlog::error(status.message());
            return;
        }
        nlohmann::json checkpoint_json = last_checkpoint(version_to_checkpoint, 3013);

        auto checkpoint_path = "/_deeplake_log/_last_checkpoint.json";
        storage_->set_bytes(checkpoint_path, checkpoint_json.dump());
    }

    arrow::Result<std::shared_ptr<arrow::Table>> deeplog::read_checkpoint(const std::string &dir_path, const long &version) const {
        arrow::MemoryPool *pool = arrow::default_memory_pool();
        auto input = open_arrow_istream(storage_->file(dir_path + "/" + zero_pad(version) + ".checkpoint.parquet"));

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

        std::shared_ptr<arrow::io::BufferOutputStream> outfile;
        ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::BufferOutputStream::Create());
//
        ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*checkpoint_table, arrow::default_memory_pool(), outfile, 3, props, arrow_props));

        auto buffer = outfile->Finish().ValueOrDie();
        const uint8_t *byte_data = buffer->data();
        int64_t byte_data_size = buffer->size();

        std::stringstream outstream;
        // Print the byte array
        for (auto i = 0; i < byte_data_size; ++i) {
            outstream << byte_data[i];
        }

        storage_->set_bytes("/_deeplake_log/" + zero_pad(version) + ".checkpoint.parquet", outstream.str());

        return arrow::Status::OK();
    }

    std::shared_ptr<arrow::io::RandomAccessFile> deeplog::open_arrow_istream(const storage::file_ref &file) const {
        auto file_data = storage_->get_bytes(file.path);
        std::string file_str = std::string(file_data.begin(), file_data.end());

        auto buffer = arrow::Buffer::FromString(file_str);

        return arrow::Buffer::GetReader(buffer).ValueOrDie();
    }

} // deeplake