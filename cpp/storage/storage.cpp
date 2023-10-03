#include "storage.hpp"

namespace storage {

    bool file_ref::exists() {
        return size >= 0;
    }
}