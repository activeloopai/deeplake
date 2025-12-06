#pragma once

/**
 * @file writer.hpp
 * @brief Definition of `writer` class.
 */

#include "exceptions.hpp"
#include "provider_base.hpp"
#include "resource_meta.hpp"

#include <async/promise.hpp>
#include <base/read_buffer.hpp>

#include <exception>
#include <functional>
#include <string>

namespace storage {

class reader;

class writer : public virtual provider_base
{
public:
    virtual ~writer() noexcept override = default;

    /**
     * @brief Create reader object from the current writer to it can be passed to the read only places
     *
     * @return std::shared_ptr<reader>
     */
    [[nodiscard]] virtual std::shared_ptr<reader> get_reader() const noexcept = 0;

    /**
     * @brief Generic interface to write data to the storage
     *
     * @param bucket the bucket/blob where the data need to be added/appended
     * @param buffer the data that needs to be added
     * @return async::promise<bool>
     */
    [[nodiscard]] virtual async::promise<resource_meta> write(const std::string& bucket,
                                                              const base::read_buffer& buffer) = 0;

    /**
     * @brief Returns writer for the given subpath of this writer.
     *
     * @param subpath Subpath.
     * @return std::shared_ptr<writer> New writer.
     */
    [[nodiscard]] virtual std::shared_ptr<writer> writer_for_subpath(const std::string& subpath) const noexcept = 0;

    /**
     * @brief Generic interface to remove the path from the storage
     *
     * @param path the path that needs to be deleted
     * @return async::promise<bool>
     */
    [[nodiscard]] virtual async::promise<bool> remove(const std::string& path) const = 0;

    /**
     * @brief Generic interface to remove the path from the storage
     *
     * @param path the path that needs to be deleted
     * @return async::promise<void>
     */
    [[nodiscard]] async::promise<void> remove_with_exception(const std::string& path) const
    {
        return remove(path).then([p = path](bool ok) {
            if (!ok) {
                throw write_exception("Failed to remove file " + p);
            }
        });
    }

    /**
     * @brief Generic interface to check if the path exists in the storage
     *
     * @param path the path that needs to be checked
     * @return async::promise<bool>
     */
    [[nodiscard]] virtual async::promise<bool> remove_directory(const std::string& path) const = 0;
};

} // namespace storage
