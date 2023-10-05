#include "util.hpp"
#include <uuid.h>

namespace deeplog {
    std::string generate_id() {
        std::random_device rd;
        auto seed_data = std::array<int, std::mt19937::state_size>{};
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        std::mt19937 generator(seq);

        auto val = uuids::to_string(uuids::uuid_random_generator{generator}());
        val.erase(std::remove(val.begin(), val.end(), '-'), val.end());

        return val;
    }

    long current_timestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
