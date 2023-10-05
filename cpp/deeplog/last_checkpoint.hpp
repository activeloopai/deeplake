#pragma once

#include <nlohmann/json.hpp>

namespace deeplog {

    struct last_checkpoint {
    public:

        last_checkpoint();

        last_checkpoint(unsigned long version, long size);

        unsigned long version;
        long size;
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(last_checkpoint, version, size);
}
