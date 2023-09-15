#include <gtest/gtest.h>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "add_file_action.hpp"

using json = nlohmann::json;

TEST(AddFileActionTest, to_json) {
    auto action = deeplake::add_file_action("my/path", 3, 551, true);

    json j = json::object();
    action.to_json(j);
    EXPECT_EQ("{\"add\":{\"dataChange\":true,\"modificationTime\":551,\"path\":\"my/path\",\"size\":3}}", j.dump());

    auto parsed = deeplake::add_file_action(j);
    EXPECT_EQ("my/path", parsed.path());
    EXPECT_EQ(3, parsed.size());
    EXPECT_EQ(551, parsed.modification_time());
}