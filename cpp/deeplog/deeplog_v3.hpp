#pragma once


#include "deeplog.hpp"

namespace deeplog {
class deeplog_v3 : public deeplog {
    public:
        deeplog_v3(const std::string &path);
        int log_format() const override;
    };
}