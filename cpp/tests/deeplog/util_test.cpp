#include <gtest/gtest.h>
#include "../../deeplog/util.hpp"

TEST(UtilTest, generate_id) {
    auto id = deeplog::generate_id();
    EXPECT_FALSE(id.empty());
    EXPECT_TRUE(id.find_first_of('-') == -1);
}