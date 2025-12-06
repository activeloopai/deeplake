#pragma once

#include <nd/io.hpp>
#include <storage/resource_meta.hpp>

namespace deeplake_core {

struct datafile_header
{
    storage::resource_meta meta_;
    nd::header_info hinfo_;
};

} /// deeplake_core namespace
