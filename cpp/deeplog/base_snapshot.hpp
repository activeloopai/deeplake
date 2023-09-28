#pragma once

#include "deeplog.hpp"

namespace deeplog {
    class base_snapshot {

    public:
        long version;
        const std::shared_ptr<deeplog> deeplog;

    protected:
        std::shared_ptr<std::vector<std::shared_ptr<action>>> actions_;

        base_snapshot(const std::shared_ptr<::deeplog::deeplog> &deeplog);

        base_snapshot(const long &version, const std::shared_ptr<::deeplog::deeplog> &deeplog);

        template<typename T>
        std::vector<std::shared_ptr<T>> find_actions() const;

        template<typename T>
        std::shared_ptr<T> find_action() const;
    };
}