#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../../deeplog/actions/add_file_action.hpp"
#include "../../../deeplog/actions/remove_file_action.hpp"

using json = nlohmann::json;

TEST(RemoveFileActionTest, to_json) {
    auto action = deeplog::remove_file_action("my/path", 3, 551, true);

    json j = action.to_json();
    EXPECT_EQ("{\"dataChange\":true,\"deletionTime\":551,\"path\":\"my/path\",\"size\":3}", j.dump());
}