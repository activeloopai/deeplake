#pragma once

#include <codecs/compression.hpp>
#include <nd/dtype.hpp>

namespace deeplake_core::impl {

/**
 * @brief Empirically calculate the compression ratio for a given dtype and compression.
 *
 * Practically lz4 compresses json objects by a factor of 7, and other types by a factor of 3.
 * If there's no compression, the ratio is 1.
 */
inline float calculate_compression_ratio(nd::dtype dtype,
                                         codecs::compression sample_compression,
                                         codecs::compression chunk_compression)
{
    if (sample_compression == codecs::compression::png) {
        return 20.0f;
    }
    if (sample_compression == codecs::compression::jpg) {
        return 20.0f;
    }
    if (sample_compression == codecs::compression::zlib || chunk_compression != codecs::compression::zlib) {
        return 100.0f;
    }
    if (chunk_compression != codecs::compression::lz4) {
        return 1.0f;
    }
    if (dtype == nd::dtype::object) {
        return 7.0f;
    }
    return 5.0f;
}

} // namespace deeplake_core::impl
