#pragma once

#include <base/base.hpp>

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <string>
#include <utility>

namespace base {

enum class htype : unsigned char
{
    image3d,
    nifti,
    dicom,
    image,
    video,
    audio,
    binary_mask,
    segment_mask,
    bbox,
    bbox3d,
    point,
    point_cloud,
    keypoint,
    keypoint3d_box,
    polygon,
    intrinsics,
    embedding,
    label,
    text,
    generic,
    mesh,

    tag,
    json,
    list,
    medical,
    struct_,
    unknown,
};

bool htype_is_biomedical(htype t);

htype htype_from_string(const std::string& j);

template<typename T>
htype htype_from_json(const T& j)
{
    if (j.is_null()) {
        base::log_warning(log_channel::generic, "null htype");
        return htype::unknown;
    }
    auto s = j.is_string() ? j.template get<std::string>() : std::string();

    return htype_from_string(s);
}

std::string htype_to_str(htype t);

std::pair<bool, std::string> get_is_sequence_and_htype(const std::string& htype);

}
