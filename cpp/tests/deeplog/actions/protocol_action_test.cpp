#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../../deeplog/actions/protocol_action.hpp"
#include <iostream>
#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/json/api.h>

using json = nlohmann::json;


//TEST(ActionTest, JsonTest) {
//    // Your JSON string (replace with your actual JSON data)
//    const std::string json_data = R"(
//        {"name": "Alice", "age": 30}
//        {"name": "Bob", "age": 25}
//        {"name": "Charlie", "age": 35}
//    )";
//
//    // Create an Arrow Memory Pool
//    auto memory_pool = arrow::default_memory_pool();
//
//    // Create an Arrow Buffer from the JSON string
//    auto json_buffer = arrow::Buffer::FromString(json_data);
//
//    // Create an Arrow BufferReader from the JSON buffer
//    std::shared_ptr<arrow::io::BufferReader> buffer_reader = std::make_shared<arrow::io::BufferReader>(json_buffer);
//
//    auto read_options = arrow::json::ReadOptions::Defaults();
//    auto parse_options = arrow::json::ParseOptions::Defaults();
//    parse_options.explicit_schema = arrow::schema({
//                                                         arrow::field("name", arrow::int32()),
//                                                         arrow::field("age", arrow::int32()),
//                                                 });
//
//    // Create a JSON reader
//    std::shared_ptr<arrow::json::TableReader> json_reader;
//    auto status= arrow::json::TableReader::Make(memory_pool, buffer_reader, read_options, parse_options);
//    if (!status.ok()) {
//        std::cerr << "JSON reader creation failed: " << status.status() << std::endl;
//        return;
//    }
//    json_reader = status.ValueOrDie();
//
//    // Read the JSON data into an Arrow Table
//    std::shared_ptr<arrow::Table> arrow_table;
//    auto reader_status = json_reader->Read();
//    if (!reader_status.ok()) {
//        std::cerr << "JSON read failed: " << reader_status.status() << std::endl;
//        return;
//    }
//    arrow_table = reader_status.ValueOrDie();
//
//    // Print the Arrow Table for verification
//    std::cout << "Arrow Table:\n" << arrow_table->ToString() << std::endl;
//
//    arrow_table->
//}

TEST(ActionTest, UpgradeProtocolJson) {

    auto orig = deeplog::protocol_action(5, 6);

    json j = json::object();
    orig.to_json(j);

    EXPECT_EQ("{\"protocol\":{\"minReaderVersion\":5,\"minWriterVersion\":6}}", j.dump());

    auto parsed = deeplog::protocol_action(j);
    EXPECT_EQ(5, parsed.min_reader_version);
    EXPECT_EQ(6, parsed.min_writer_version);
}

