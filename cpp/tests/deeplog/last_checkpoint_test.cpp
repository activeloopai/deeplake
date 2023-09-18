#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../deeplog/last_checkpoint.hpp"

TEST(LastCheckpoint, to_json) {
    nlohmann::json j = deeplog::last_checkpoint(31, 1003);
    EXPECT_EQ("{\"size\":1003,\"version\":31}", j.dump());

    auto parsed = j.template get<deeplog::last_checkpoint>();
    EXPECT_EQ(1003, parsed.size);
    EXPECT_EQ(31, parsed.version);
}