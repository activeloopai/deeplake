#pragma once

#include "datafile_header.hpp"

#include <async/promise.hpp>
#include <format/buffer.hpp>
#include <nd/array.hpp>

#include <memory>

namespace storage {
class writer;
} // namespace storage

namespace deeplake_core {

class datafile_writer : public std::enable_shared_from_this<datafile_writer>
{
public:
    explicit datafile_writer(std::shared_ptr<storage::writer> writer, std::string path_prefix = std::string());
    virtual ~datafile_writer() = default;
    virtual std::string construct_filename() const;

    [[nodiscard]] async::promise<datafile_header> write(const nd::array& data) const;

protected:
    using write_to_buffer_result_t = std::pair<format::buffer_t, nd::header_info>;
    [[nodiscard]] virtual async::promise<write_to_buffer_result_t> write_to_buffer(const nd::array& data) const = 0;

private:
    std::shared_ptr<storage::writer> writer_;
    std::string path_prefix_;
};

} // namespace deeplake_core
