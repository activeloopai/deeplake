#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <arrow/io/memory.h>
#include <arrow/json/api.h>
#include <arrow/type.h>
#include "../../../deeplog/actions/action.hpp"
#include "../../../deeplog/actions/add_file_action.hpp"
#include "../../../deeplog/actions/create_branch_action.hpp"
#include "../../../deeplog/actions/create_commit_action.hpp"
#include "../../../deeplog/actions/create_tensor_action.hpp"
#include "../../../deeplog/actions/metadata_action.hpp"
#include "../../../deeplog/actions/protocol_action.hpp"
#include "../../../deeplog/actions/remove_file_action.hpp"

enum ActionDataContentsType {
    FILLED,
    NULLS,
    NEGATIVES,
    ZEROS,
};

template<typename ActionType>
class ActionSerializationTest : public testing::Test {
public:
    void test_data_conversions(ActionDataContentsType test_type) {
        auto original_action_json = nlohmann::json::object();
        for (const auto &field: ActionType::arrow_type->fields()) {
            if (test_type == NULLS && field->nullable()) {
                original_action_json[field->name()] = nlohmann::json::value_t::null;
                continue;
            }

            if (field->type()->Equals(arrow::utf8())) {
                original_action_json[field->name()] = "my/path";
            } else if (field->type()->Equals(arrow::uint64())) {
                if (test_type == ZEROS) {
                    original_action_json[field->name()] = 0;
                } else {
                    original_action_json[field->name()] = 873731;
                }
            } else if (field->type()->Equals(arrow::int64()) || field->type()->Equals(arrow::int32())) {
                if (test_type == ZEROS) {
                    original_action_json[field->name()] = 0;
                } else if (test_type == NEGATIVES) {
                    original_action_json[field->name()] = -54812;
                } else {
                    original_action_json[field->name()] = 873731;
                }
            } else if (field->type()->Equals(arrow::boolean())) {
                if (test_type == NEGATIVES) {
                    original_action_json[field->name()] = false;
                } else {
                    original_action_json[field->name()] = true;
                }
            } else if (field->type()->Equals(arrow::map(arrow::utf8(), arrow::map(arrow::utf8(), arrow::utf8())))) {
                std::map<std::string, std::map<std::string, std::string>> data = {
                        {"outer1", {
                                {"key1", "val1"},
                                {"key2", "val2"},
                        }
                        },
                        {"outer2", {
                                {"key3", "val3"},
                                {"key4", "val4"},
                        }
                        },
                };

                auto auto_data1 = std::map<std::string, std::string>();

                original_action_json[field->name()] = data;
            } else if (field->type()->Equals(arrow::list(arrow::uint64()))) {
                std::vector<unsigned long> data = {1,
                                                   5,
                                                   13131};

                auto auto_data1 = std::map<std::string, std::string>();

                original_action_json[field->name()] = data;
            } else {
                throw std::runtime_error("No test data generation configured for type " + field->type()->ToString());
            }
        }

        auto table_json = "{\"x\":" + original_action_json.dump() + "}";
        auto input = std::make_shared<arrow::io::BufferReader>(std::make_shared<arrow::Buffer>(table_json));

        auto parse_options = arrow::json::ParseOptions::Defaults();
        parse_options.explicit_schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
                arrow::field("x", ActionType::arrow_type),
        });

        auto table = arrow::json::TableReader::Make(arrow::default_memory_pool(), input, arrow::json::ReadOptions::Defaults(),
                                                    parse_options).ValueOrDie()->Read().ValueOrDie();

        auto parsed_scalar = std::dynamic_pointer_cast<arrow::StructScalar>(table->GetColumnByName("x")->chunk(0)->GetScalar(0).ValueOrDie());

        auto from_arrow_action = ActionType(parsed_scalar);

        EXPECT_EQ(original_action_json.dump(), from_arrow_action.to_json().dump());
    }
};


using ActionTypes = ::testing::Types<
        deeplog::add_file_action,
        deeplog::create_branch_action,
        deeplog::create_commit_action,
        deeplog::create_tensor_action,
        deeplog::metadata_action,
        deeplog::protocol_action,
        deeplog::remove_file_action
>;
TYPED_TEST_SUITE(ActionSerializationTest, ActionTypes);


TYPED_TEST(ActionSerializationTest, FilledData) {
    this->test_data_conversions(FILLED);
}

//TYPED_TEST(ActionSerializationTest, NullValues) {
//    this->test_data_conversions(NULLS);
//}

TYPED_TEST(ActionSerializationTest, NegativeValues) {
    this->test_data_conversions(NEGATIVES);
}

TYPED_TEST(ActionSerializationTest, ZeroValues) {
    this->test_data_conversions(ZEROS);
}

//class ActionDataHandlingTest : public ::testing::TestWithParam<std::tuple<ActionDataContentsType, std::shared_ptr<ActionDataHandler>>> {
//
//};

//TEST_P(ActionDataHandlingTest, auto_test) {
//    auto [test_type, handler] = GetParam();
//
//    auto original_action_json = nlohmann::json::object();
//    for (auto field: handler->arrow_type()->fields()) {
//        if (test_type == NULLS && field->nullable()) {
//            original_action_json[field->name()] = nlohmann::json::value_t::null;
//            continue;
//        }
//
//        if (field->type()->Equals(arrow::utf8())) {
//            original_action_json[field->name()] = "my/path";
//        } else if (field->type()->Equals(arrow::uint64())) {
//            if (test_type == ZEROS) {
//                original_action_json[field->name()] = 0;
//            } else {
//                original_action_json[field->name()] = 873731;
//            }
//        } else if (field->type()->Equals(arrow::boolean())) {
//            if (test_type == NEGATIVES) {
//                original_action_json[field->name()] = false;
//            } else {
//                original_action_json[field->name()] = true;
//            }
//        } else {
//            throw std::runtime_error("Unsupported type " + field->type()->ToString());
//        }
//    }
//
//    auto table_json = "{\"x\":" + original_action_json.dump() + "}";
//    auto input = std::make_shared<arrow::io::BufferReader>(std::make_shared<arrow::Buffer>(table_json));
//
//    auto parse_options = arrow::json::ParseOptions::Defaults();
//    parse_options.explicit_schema = std::make_shared<arrow::Schema>(arrow::FieldVector{
//            arrow::field("x", handler->arrow_type()),
//    });
//
//    auto table = arrow::json::TableReader::Make(arrow::default_memory_pool(), input, arrow::json::ReadOptions::Defaults(),
//                                                parse_options).ValueOrDie()->Read().ValueOrDie();
//
//    auto parsed_scalar = std::dynamic_pointer_cast<arrow::StructScalar>(table->GetColumnByName("x")->chunk(0)->GetScalar(0).ValueOrDie());
//
//    auto from_arrow_action = handler->create_action(parsed_scalar);
//
//    EXPECT_EQ(original_action_json.dump(), from_arrow_action->to_json().dump());
//}