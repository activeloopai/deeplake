#pragma once

#include "deeplog.hpp"

namespace deeplog {

    class metadata_snapshot {
    public:
        metadata_snapshot(const std::shared_ptr<deeplog> &deeplog);

        metadata_snapshot(const long &version, const std::shared_ptr<deeplog> &deeplog);

        const long version;

        std::shared_ptr<protocol_action> protocol() const;

        std::shared_ptr<metadata_action> metadata() const;

        std::vector<std::shared_ptr<create_branch_action>> branches() const;

        std::shared_ptr<create_branch_action> branch_by_id(const std::string &branch_id) const;

        std::optional<std::string> branch_id(const std::string &name) const;

    private:
        std::shared_ptr<deeplog> deeplog_;
    };

}