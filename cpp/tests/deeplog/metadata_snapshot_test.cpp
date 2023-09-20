#include <gtest/gtest.h>
#include <filesystem>
#include "../../deeplog/deeplog.hpp"
#include "../../deeplog/metadata_snapshot.hpp"

class DeeplogMetadataSnapshotTest : public ::testing::Test {
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

TEST_F(DeeplogMetadataSnapshotTest, construct) {
    auto log = deeplog::deeplog::create(test_dir);

    auto original_metadata = log->metadata().data;
    auto action = deeplog::metadata_action(original_metadata->id, "new name", "new desc", original_metadata->created_time);
    log->commit(deeplog::MAIN_BRANCH_ID, log->version(deeplog::MAIN_BRANCH_ID), {std::make_shared<deeplog::metadata_action>(action)});

    auto snapshot0 = deeplog::metadata_snapshot(0, log);
    auto snapshot1 = deeplog::metadata_snapshot(1, log);

    EXPECT_EQ(0, snapshot0.version);
    EXPECT_EQ(1, snapshot1.version);
}