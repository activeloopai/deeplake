#pragma once

#include <heimdall/dataset_view.hpp>

namespace deeplake {
class dataset_view;
class tag;
} // namespace deeplake

namespace deeplake_api {

class read_only_dataset;

class tag_view
{
public:
    tag_view(const std::shared_ptr<deeplake::tag>& tag, const std::shared_ptr<deeplake::dataset_view>& dataset)
        : base_tag_(tag)
        , dataset_(dataset)
    {
    }

    [[nodiscard]] const std::string& id() const;
    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] const std::string& branch_id() const;
    [[nodiscard]] const std::string& version() const;
    [[nodiscard]] const std::chrono::system_clock::time_point& timestamp() const;
    [[nodiscard]] const std::string& message() const;
    [[nodiscard]] const auto& filter() const;
    [[nodiscard]] const std::string& to_string() const;
    [[nodiscard]] async::promise<heimdall::dataset_view_ptr> open();

    // Link support
    [[nodiscard]] bool is_link() const;
    [[nodiscard]] const std::optional<std::string>& link_target() const;

private:
    [[nodiscard]] async::promise<heimdall::dataset_view_ptr>
    apply_filter(async::promise<std::shared_ptr<read_only_dataset>>&& promise);

protected:
    std::shared_ptr<deeplake::tag> base_tag_;
    std::shared_ptr<deeplake::dataset_view> dataset_;
};

} // namespace deeplake_api
