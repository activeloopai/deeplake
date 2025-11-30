#pragma once

#include "tag_view.hpp"

namespace deeplake {
class dataset_view;
} // namespace deeplake

namespace deeplake_api {

class tags_view
{
public:
    explicit tags_view(const std::shared_ptr<deeplake::dataset_view>& dataset)
        : dataset_(dataset)
    {
    }

    [[nodiscard]] int64_t size() const;
    [[nodiscard]] tag_view at(const std::string& name_or_id) const;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::vector<std::string> names() const;

private:
    std::shared_ptr<deeplake::dataset_view> dataset_;
};

}
