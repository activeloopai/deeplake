#ifndef DEEPLAKE_LAST_CHECKPOINT_HPP
#define DEEPLAKE_LAST_CHECKPOINT_HPP

#include <nlohmann/json.hpp>

namespace deeplake {

    struct last_checkpoint {
    public:

        last_checkpoint();

        last_checkpoint(long version, long size);

        long version;
        long size;
    };

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(last_checkpoint, version, size);
}


#endif //DEEPLAKE_LAST_CHECKPOINT_HPP
