#include <gtest/gtest.h>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "create_branch_action.hpp"

using json = nlohmann::json;

TEST(CreateBranchActionTest, to_json) {
    auto action = deeplake::create_branch_action("my_id", "my_name", "other_id", 93);

    json j = json::object();
    action.to_json(j);
    EXPECT_EQ("{\"createBranch\":{\"fromBranchId\":\"other_id\",\"fromVersion\":93,\"id\":\"my_id\",\"name\":\"my_name\"}}", j.dump());

    auto parsed = deeplake::create_branch_action(j);
    EXPECT_EQ("my_id", parsed.id());
    EXPECT_EQ("my_name", parsed.name());
    EXPECT_EQ("other_id", parsed.from_id());
    EXPECT_EQ(93, parsed.from_version());
}