#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../../deeplog/actions/metadata_action.hpp"

using json = nlohmann::json;

TEST(MetadataActionTest, to_json) {
    auto action = deeplog::metadata_action("asdf", "name here", "desc here", 12345);

    json j = action.to_json();
    EXPECT_EQ(
            "{\"createdTime\":12345,\"description\":\"desc here\",\"id\":\"asdf\",\"name\":\"name here\"}",
            j.dump());

    action = deeplog::metadata_action("asdf", std::nullopt, std::nullopt, 12345);
    j = action.to_json();
    EXPECT_EQ("{\"createdTime\":12345,\"description\":null,\"id\":\"asdf\",\"name\":null}", j.dump());
}