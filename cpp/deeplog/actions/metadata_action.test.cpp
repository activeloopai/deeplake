#include <gtest/gtest.h>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "metadata_action.hpp"

using json = nlohmann::json;

TEST(MetadataActionTest, to_json) {
    auto action = deeplake::metadata_action("asdf", "name here", "desc here", 12345);

    json j = json::object();
    action.to_json(j);
    EXPECT_EQ(
            "{\"metadata\":{\"createdTime\":12345,\"description\":\"desc here\",\"id\":\"asdf\",\"name\":\"name here\"}}",
            j.dump());

    auto parsed = deeplake::metadata_action(j);
    EXPECT_EQ("asdf", parsed.id());
    EXPECT_EQ("name here", parsed.name());
    EXPECT_EQ("desc here", parsed.description());
    EXPECT_EQ(12345, parsed.created_time());


    action = deeplake::metadata_action("asdf", std::nullopt, std::nullopt, 12345);
    j = json::object();
    action.to_json(j);
    EXPECT_EQ("{\"metadata\":{\"createdTime\":12345,\"description\":null,\"id\":\"asdf\",\"name\":null}}", j.dump());

    parsed = deeplake::metadata_action(j);
    EXPECT_EQ("asdf", parsed.id());
    EXPECT_FALSE(parsed.name().has_value());
    EXPECT_FALSE(parsed.description().has_value());
    EXPECT_EQ(12345, action.created_time());
}