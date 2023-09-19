#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../../deeplog/actions/create_tensor_action.hpp"

using json = nlohmann::json;

TEST(CreateTensorActionTest, to_json) {

    auto action = deeplog::create_tensor_action("my_id", "my_name", "my_dtype", "my_htype", 93,
                                                true, false, true, "my_chunk_compression", "my_sample_compression",
                                                std::map<std::string, std::map<std::string, std::string>>(),
                                                93,
                                                {}, {}, 53, "my_typestr", true, "my_version");

    json j = json::object();
    action.to_json(j);
    EXPECT_EQ("{\"tensor\":{\"chunkCompression\":\"my_chunk_compression\",\"dtype\":\"my_dtype\",\"hidden\":true,\"htype\":\"my_htype\",\"id\":\"my_id\",\"length\":93,\"link\":true,\"links\":{},\"maxChunkSize\":93,\"maxShape\":[],\"minShape\":[],\"name\":\"my_name\",\"sampleCompression\":\"my_sample_compression\",\"sequence\":false,\"tilingThreshold\":53,\"typestr\":\"my_typestr\",\"verify\":true,\"version\":\"my_version\"}}", j.dump());

    auto parsed = deeplog::create_tensor_action(j);
    EXPECT_EQ("my_id", parsed.id());
    EXPECT_EQ("my_name", parsed.name());
}