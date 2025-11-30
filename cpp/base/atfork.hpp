#pragma once

#include <any>
#include <memory>

namespace base {

struct atfork_handler : public std::enable_shared_from_this<atfork_handler>
{
    atfork_handler() = default;
    virtual ~atfork_handler() = default;

    virtual std::any before_fork() const
    {
        return std::any();
    }

    virtual void after_fork_parent(std::any token)
    {
    }

    virtual void after_fork_child(std::any token)
    {
    }
};

void register_atfork_handler(std::weak_ptr<atfork_handler> weak_handler);

} // namespace base