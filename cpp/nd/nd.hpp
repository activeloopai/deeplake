#pragma once

/**
 * @defgroup nd
 * @{
 * @brief nd module.
 * 
 * Multidimensional array class and operations on top of it.
 *
 * @}
 */

namespace nd {

class nd
{
public:
    static nd& instance()
    {
        static nd instance_;
        return instance_;
    }
};

} // namespace nd
