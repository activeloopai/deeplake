#include "last_checkpoint.hpp"

namespace deeplog {

    last_checkpoint::last_checkpoint() : version(-1), size(0) {}
    last_checkpoint::last_checkpoint(long version, long size) : version(version), size(size) {}

}