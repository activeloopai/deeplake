#pragma once

#include "snapshot.hpp"

namespace deeplog {
    class optimistic_transaction {
    public:
        optimistic_transaction(const std::shared_ptr<base_snapshot> &snapshot);

    public:
        std::shared_ptr<base_snapshot> snapshot;

    public:
        void add(const std::shared_ptr<action> &action);

        long commit();

    private:
        std::vector<std::shared_ptr<action>> actions_;
    };

}