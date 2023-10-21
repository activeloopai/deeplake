#include "deeplog_v3.hpp"

namespace deeplog {

    deeplog_v3::deeplog_v3(const std::shared_ptr<storage::storage> &storage) : deeplog(storage){}

    int deeplog_v3::log_format() const {
        return 3;
    }
}