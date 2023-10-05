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
    auto log = deeplog::deeplog::create(test_dir, 4);
    auto main_id = deeplog::metadata_snapshot(log).find_branch("main")->id;

    auto action = deeplog::add_file_action("my/path", "chunk", 3, 45, true, 20);
    log->commit(main_id, log->version(main_id), {std::make_shared<deeplog::add_file_action>(action)});

    auto snapshot0 = deeplog::snapshot(main_id, 0, log);
    auto snapshot1 = deeplog::snapshot(main_id, 1, log);

    EXPECT_EQ(0, snapshot0.version);
    EXPECT_EQ(1, snapshot1.version);

    EXPECT_EQ(main_id, snapshot0.branch_id);
    EXPECT_EQ(main_id, snapshot1.branch_id);
}
