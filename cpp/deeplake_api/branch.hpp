#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace async {
template <typename T>
class promise;
} // namespace async

namespace deeplake {
class branch;
} // namespace deeplake

namespace deeplake_api {

class branch_view
{
public:
    explicit branch_view(std::shared_ptr<deeplake::branch> impl)
        : impl_(std::move(impl))
    {
    }

    const auto& get_impl() const
    {
        return impl_;
    }

    [[nodiscard]] async::promise<bool> is_main() const;
    [[nodiscard]] async::promise<std::string> get_name() const;
    [[nodiscard]] async::promise<std::string> get_id() const;
    [[nodiscard]] async::promise<std::chrono::system_clock::time_point> get_created_time() const;
    [[nodiscard]] async::promise<std::optional<std::pair<std::string, std::string>>> get_parent_str() const;

    bool is_link() const;
    std::optional<std::string> link_target() const;

    bool operator==(const branch_view& other) const = default;

private:
    std::shared_ptr<deeplake::branch> impl_;
};

class branch : public branch_view
{
public:
    branch(std::shared_ptr<deeplake::branch> impl)
        : branch_view(std::move(impl))
    {
    }

    [[nodiscard]] async::promise<void> rename(std::string new_name);
    [[nodiscard]] async::promise<void> del();
};

} // namespace deeplake_api
