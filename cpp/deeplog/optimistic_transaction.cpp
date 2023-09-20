#include "optimistic_transaction.hpp"
#include <iostream>

namespace deeplog {
    optimistic_transaction::optimistic_transaction(const std::shared_ptr<::deeplog::snapshot> &snapshot) : snapshot(snapshot), actions_({}) {}

    void optimistic_transaction::add(const std::shared_ptr<action> &action) {
        actions_.push_back(action);
    }

    long optimistic_transaction::commit() {
        snapshot->deeplog->commit(snapshot->branch_id, snapshot->version, actions_);
        return 3;
    }
}