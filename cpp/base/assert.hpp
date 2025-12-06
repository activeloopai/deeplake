#pragma once

/**
 * @file assert.hpp
 * @brief Macro for indra specific ASSERT.
 */

#ifdef AL_ASSERTIONS

#include "backtrace.hpp"

#include <string>

namespace base {

[[noreturn]] void abort(const std::string& message);

}

#define ASSERT_MESSAGE(expression, message)                                                                            \
    if (!(expression))                                                                                                 \
    base::abort(std::string("Assertion Failed: ") + #expression + "\nMessage: " + message + "\nFile: " + __FILE__ +    \
                ":" + std::to_string(__LINE__) + "\nBacktrace:\n" + base::backtrace())

#else // AL_ASSERTIONS

#define ASSERT_MESSAGE(expression, message) ((void)0)

#endif // AL_ASSERTIONS

#define ASSERT(expression) ASSERT_MESSAGE((expression), #expression)
