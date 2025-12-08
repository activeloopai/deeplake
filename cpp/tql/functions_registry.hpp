#pragma once

#include "function_variant.hpp"

#include <async/promise.hpp>

#include <string>
#include <vector>

namespace tql {

/**
 * @class functions_registry
 * @brief Abstract registry to define and expose custom functions to the TQL runtime.
 */
class functions_registry
{
public:
    virtual ~functions_registry() = default;

public:
    /**
     * @brief Returns names list of all functions in the registry.
     */
    virtual std::vector<std::string> functions_names() const = 0;

    /**
     * @brief Returns the real functions which needs to be executed.
     */
    virtual async::promise<function_variant> get_function(const std::string& name, int arguments_count) = 0;
};

}
