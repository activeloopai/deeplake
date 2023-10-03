#pragma once


#include "deeplog.hpp"

namespace deeplog {
class deeplog_v3 : public deeplog {
    public:
        deeplog_v3(const std::shared_ptr<storage::storage> &storage);
        int log_format() const override;
    };
}