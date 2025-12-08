#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace async {
template <typename T>
class promise;
} // namespace async

namespace deeplake {
class branch;
class dataset_view;
} // namespace deeplake

namespace deeplog {
class commit;
} // namespace deeplog

namespace deeplake_api {

class read_only_dataset;

class version
{
public:
    version(std::shared_ptr<const deeplog::commit> commit,
            std::shared_ptr<deeplake::dataset_view> dataset,
            std::shared_ptr<deeplake::branch> branch);

    [[nodiscard]] const std::string& id() const;

    /**
     * @brief The epoch time the user's machine had when committing
     */
    [[nodiscard]] std::chrono::system_clock::time_point client_timestamp() const;

    /**
     * @brief The epoch time the commit based on the storage's clock
     */
    [[nodiscard]] std::chrono::system_clock::time_point timestamp() const;

    [[nodiscard]] const std::optional<std::string>& message() const;

    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] async::promise<std::shared_ptr<read_only_dataset>> open() const;

private:
    std::shared_ptr<const deeplog::commit> commit_;
    std::shared_ptr<deeplake::dataset_view> dataset_;
    std::shared_ptr<deeplake::branch> branch_;
};

} // namespace deeplake_api
