#pragma once

#include "tag_view.hpp"

namespace deeplake_api {

class tag : public tag_view
{
public:
    tag(const std::shared_ptr<deeplake::tag>& tag, const std::shared_ptr<deeplake::dataset_view>& dataset)
        : tag_view(tag, dataset)
    {
    }

    [[nodiscard]] async::promise<void> rename(const std::string& new_name);
    [[nodiscard]] async::promise<void> del();
};

} // namespace deeplake_api
