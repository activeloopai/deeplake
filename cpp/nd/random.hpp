#pragma once

#include "array.hpp"
#include "dtype.hpp"

namespace nd {

/**
 * @brief Generate a random array with the given data type and shape.
 * 
 * @param t The data type of the array.
 * @param shape The shape of the array.
 * @return array The generated random array.
 */
array random(dtype t, icm::shape shape);

}
