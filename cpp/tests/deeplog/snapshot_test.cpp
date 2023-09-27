#include <gtest/gtest.h>
#include <filesystem>
#include "../../deeplog/deeplog.hpp"
#include "../../deeplog/snapshot.hpp"
#include "../../deeplog/metadata_snapshot.hpp"

class DeeplogSnapshotTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    void TearDown() override {
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    std::string test_dir = "tmp/test";
};

TEST_F(DeeplogSnapshotTest, construct) {
    auto log = deeplog::deeplog::create(test_dir);

    auto original_metadata = deeplog::metadata_snapshot(log).metadata();
    auto action = deeplog::metadata_action(original_metadata->id, "new name", "new desc", original_metadata->created_time);
    log->commit(deeplog::MAIN_BRANCH_ID, log->version(deeplog::MAIN_BRANCH_ID), {std::make_shared<deeplog::metadata_action>(action)});

    auto snapshot0 = deeplog::snapshot(deeplog::MAIN_BRANCH_ID, 0, log);
    auto snapshot1 = deeplog::snapshot(deeplog::MAIN_BRANCH_ID, 1, log);

    EXPECT_EQ(0, snapshot0.version);
    EXPECT_EQ(1, snapshot1.version);

    EXPECT_EQ(deeplog::MAIN_BRANCH_ID, snapshot0.branch_id);
    EXPECT_EQ(deeplog::MAIN_BRANCH_ID, snapshot1.branch_id);
}