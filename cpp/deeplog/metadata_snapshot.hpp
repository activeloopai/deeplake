#pragma once

#include "deeplog.hpp"
#include "base_snapshot.hpp"

namespace deeplog {

    class metadata_snapshot : public base_snapshot {
    public:
        metadata_snapshot(const std::shared_ptr<::deeplog::deeplog> &deeplog);

        metadata_snapshot(const unsigned long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog);

        std::shared_ptr<base_snapshot> update() const override;

        std::shared_ptr<protocol_action> protocol() const;

        std::shared_ptr<metadata_action> metadata() const;

        std::vector<std::shared_ptr<create_branch_action>> branches() const;

        std::shared_ptr<create_branch_action> find_branch(const std::string &address) const;

    };
}