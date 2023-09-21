project(tests)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
enable_testing()


file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests
        deeplog
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(runTests)
