#pragma once

#include <async/priority.hpp>
#include <base/memory_policy.hpp>

namespace storage {

struct fetch_options
{
    fetch_options() = default;

    explicit fetch_options(int p)
        : priority(p)
    {
    }

    explicit fetch_options(base::memory_policy p)
        : memory_policy(p)
    {
    }

    fetch_options(int pr, base::memory_policy po)
        : priority(pr)
        , memory_policy(po)
    {
    }

    const int priority = async::default_priority;
    const base::memory_policy memory_policy = base::memory_policy::aggressive;
};

}
