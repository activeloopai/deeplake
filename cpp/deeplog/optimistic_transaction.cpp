#include "optimistic_transaction.hpp"
#include "spdlog/spdlog.h"
#include <iostream>

namespace deeplog {
    optimistic_transaction::optimistic_transaction(const std::shared_ptr<::deeplog::base_snapshot> &snapshot) : snapshot(snapshot), actions_({}) {}

    void optimistic_transaction::add(const std::shared_ptr<action> &action) {
        actions_.push_back(action);
    }

    unsigned long optimistic_transaction::commit() {
        auto snapshot_to_commit = snapshot;

        while (true) {
            auto succeeded = snapshot_to_commit->deeplog->commit(snapshot->branch_id, snapshot->version, actions_);
            if (succeeded) {
                return snapshot_to_commit->version + 1;
            }

            spdlog::debug("Commit failed, retrying");
            auto snapshot_to_commit = snapshot->update();
        }

    }
}