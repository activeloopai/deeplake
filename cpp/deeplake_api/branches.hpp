#pragma once

#include "branch.hpp"

#include <memory>
#include <string>
#include <vector>

namespace deeplake {
class branches;
} // namespace deeplake

namespace deeplake_api {

class branches_view
{
public:
    explicit branches_view(const deeplake::branches& impl);

    [[nodiscard]] async::promise<std::size_t> size() const;
    [[nodiscard]] async::promise<std::string> to_string() const;
    [[nodiscard]] async::promise<std::vector<std::string>> names() const;
    [[nodiscard]] async::promise<branch_view> get(std::string name_or_id) const;

protected:
    std::shared_ptr<deeplake::branches> impl_;
};


class branches : public branches_view
{
public:
    explicit branches(const deeplake::branches& impl)
        : branches_view(impl)
    {
    }

    [[nodiscard]] async::promise<branch> get(std::string name_or_id) const;
};

}
