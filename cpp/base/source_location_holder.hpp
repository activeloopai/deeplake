#pragma once

/**
 * @file source_location_holder.hpp
 * @brief Definition of the `source_location_holder` class for tracking source locations
 */

#include <cstddef>

// Try to detect std::source_location support
#if defined(__has_include) && __has_include(<source_location>)
    #include <source_location>
    #define HAS_SOURCE_LOCATION 1
#else
    #define HAS_SOURCE_LOCATION 0
#endif

namespace base {

#if HAS_SOURCE_LOCATION

class source_location_holder
{
public:
    using source_location = std::source_location;

    source_location_holder(std::source_location location = std::source_location::current())
        : location_(location)
    {
    }

    const std::source_location& get_source_location() const
    {
        return location_;
    }

protected:
    std::source_location location_;
};

#else

class source_location_holder
{
public:
    struct source_location
    {
        const char* file_name() const
        {
            return "unknown";
        }

        const char* function_name() const
        {
            return "unknown";
        }

        unsigned int line() const
        {
            return 0;
        }

        unsigned int column() const
        {
            return 0;
        }

        static source_location current()
        {
            return source_location();
        }
    };

    source_location_holder() = default;

    source_location_holder(source_location)
    {
    }

    const source_location& get_source_location() const
    {
        static thread_local source_location loc;
        return loc;
    }
};

#endif

} // namespace base
