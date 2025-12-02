#include <utility>

namespace nd
{

class array;

std::pair<array, array> histogram(array a, int bin_count);

array bincount(array a);
array bincount(array a, int min, int max);

/**
 * @brief Returns single element array containing the mean of all elements in the input array.
 *
 * @param a
 * @return array
 */
array mean(array a);

/**
 * @brief returns single element array containing the standard deviation of all elements in the input array.
 *
 * @param a
 * @return array
 */
array stdev(array a);

/**
 * @brief returns single element array containing the median of all elements in the input array.
 *
 * @param a
 * @return array
 */
array median(array a);

} // namespace nd