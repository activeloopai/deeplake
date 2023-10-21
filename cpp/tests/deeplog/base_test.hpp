#pragma once

#include <gtest/gtest.h>
#include <filesystem>

class base_test : public ::testing::Test {
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

    std::set<std::string> list_log_files(const std::string &branch_id) {
        auto files = std::set < std::string > ();
        std::filesystem::path dir_path = {test_dir + "/_deeplake_log/" + branch_id + "/"};
        for (const auto &entry: std::filesystem::directory_iterator(dir_path)) {
            files.insert(entry.path().string().substr((test_dir + "/_deeplake_log/" + branch_id + "/").size()));
        }

        return files;
    }

    std::string test_dir = "tmp/test";
};
