#include "deeplog_v3.hpp"

namespace deeplog {

    deeplog_v3::deeplog_v3(const std::string &path) : deeplog(path) {
    }

    int deeplog_v3::log_format() const {
        return 3;
    }
}