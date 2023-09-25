#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../../deeplog/actions/create_branch_action.hpp"

using json = nlohmann::json;

TEST(CreateBranchActionTest, to_json) {
    auto action = deeplog::create_branch_action("my_id", "my_name", "other_id", 93);

    json j = action.to_json();
    EXPECT_EQ("{\"fromId\":\"other_id\",\"fromVersion\":93,\"id\":\"my_id\",\"name\":\"my_name\"}", j.dump());
}