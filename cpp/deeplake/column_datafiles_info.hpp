#pragma once

#include <string>
#include <vector>

namespace deeplake {

struct column_datafiles_info {
    struct insert_datafile_info
    {
        std::string filename;
        int64_t offset_start;
        int64_t offset_end;
    };

    struct update_datafile_info
    {
        std::string filename;
        std::vector<int64_t> offsets;
    };

    std::vector<insert_datafile_info> insert_datafiles;
    std::vector<update_datafile_info> update_datafiles;
};

} // namespace deeplake 