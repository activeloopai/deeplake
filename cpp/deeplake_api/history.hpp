#pragma once

#include "version.hpp"

#include <vector>

namespace deeplake {
class data_container;
class data_container_view;
class dataset;
class dataset_view;
} // namespace deeplake

namespace deeplake_api {

class history
{
public:
    history(const std::shared_ptr<deeplake::dataset_view>& dataset,
            const std::shared_ptr<deeplake::data_container_view>& data_container);

    history(const std::shared_ptr<deeplake::dataset>& dataset,
            const std::shared_ptr<deeplake::data_container>& data_container);

    [[nodiscard]] bool contains(const std::string& version_str) const;

    [[nodiscard]] const version& at(std::size_t index) const;

    [[nodiscard]] const version& at(std::string_view version) const;

    [[nodiscard]] std::vector<version>::const_iterator begin() const
    {
        return versions_.begin();
    }

    [[nodiscard]] std::vector<version>::const_iterator end() const
    {
        return versions_.end();
    }

    std::size_t size() const
    {
        return versions_.size();
    }

    [[nodiscard]] std::string to_string() const;

private:
    std::vector<version> versions_;
};

} // namespace deeplake_api
