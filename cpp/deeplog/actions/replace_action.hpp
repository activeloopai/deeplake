#pragma once

#include "action.hpp"

namespace deeplog {


    class replace_action {
    public:
        virtual bool replaces(std::shared_ptr<::deeplog::action> action) = 0;

        virtual std::shared_ptr<action> replace(std::shared_ptr<::deeplog::action> action) = 0;
    };
}
