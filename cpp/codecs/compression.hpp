#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace icm {
class json;
} // namespace icm

namespace codecs {

enum class compression : unsigned char
{
    null,
    png,
    apng,
    jpg,
    tiff,
    jpeg2000,
    bmp,
    nii,
    nii_gz,
    dcm,
    lz4,
    wav,
    mp3,
    mp4,
    mkv,
    avi,
    stl,
    ply,
    zlib,
    unknown
};

bool compression_is_for_biomedical_image(compression c);

compression compression_from_string(const std::string& s);

template <typename T>
compression compression_from_json(const T& j)
{
    if (j.is_null()) {
        return compression::null;
    }
    return compression_from_string(j.template get<std::string>());
}

icm::json compression_to_json(compression c);

std::string compression_to_str(compression c);

compression compression_from_file_header(std::span<const uint8_t> file);

bool is_nifti(std::span<const uint8_t> file);

// check whether the file is a valid audio file(mp3 or wav for now)
bool is_audio(std::span<const uint8_t> file);

} // namespace codecs
