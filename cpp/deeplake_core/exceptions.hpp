#pragma once

#include <base/exception.hpp>
#include <base/format.hpp>
#include <nd/dtype.hpp>

namespace deeplake_core {

class exception : public base::exception
{
public:
    explicit exception(std::string&& what)
        : base::exception(std::move(what))
    {
    }
};

class unknown_format : public exception
{
public:
    explicit unknown_format(const std::string& format)
        : exception(fmt::format("The format '{}' is unknown", format))
    {
    }
};

class invalid_chunk_strategy_type : public exception
{
public:
    explicit invalid_chunk_strategy_type(const std::string& strategy)
        : exception(fmt::format("The chunk strategy type '{}' is invalid", strategy))
    {
    }
};

class invalid_sequence_of_sequence : public exception
{
public:
    invalid_sequence_of_sequence()
        : exception("A sequence of sequence is not allowed")
    {
    }
};

class unknown_type_kind : public exception
{
public:
    explicit unknown_type_kind(const std::string& kind)
        : exception(fmt::format("The type kind '{}' is unknown", kind))
    {
    }
};

class invalid_text_type : public exception
{
public:
    invalid_text_type(nd::dtype t)
        : exception(fmt::format("The data type '{}' can't be converted to text", nd::dtype_to_str(t)))
    {
    }
};

class unsupported_sample_compression : public exception
{
public:
    unsupported_sample_compression(const std::string& comp, const std::string& supported)
        : exception(
              fmt::format("Provided sample compression '{}' is not supported. Only {} are supported", comp, supported))
    {
    }

    explicit unsupported_sample_compression(const std::string& comp)
        : exception(fmt::format("Provided sample compression '{}' is not supported", comp))
    {
    }
};

class unsupported_chunk_compression : public exception
{
public:
    explicit unsupported_chunk_compression(const std::string& comp)
        : exception(fmt::format("Provided chunk compression '{}' is not supported. Only lz4 is supported", comp))
    {
    }
};

class invalid_image_compression : public exception
{
public:
    explicit invalid_image_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided image compression '{}' is not supported. Only null, jpg and png are supported", comp))
    {
    }
};

class invalid_text_compression : public exception
{
public:
    explicit invalid_text_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided Text compression '{}' is not supported. Only null, lz4 and zstd are supported", comp))
    {
    }
};

class invalid_video_compression : public exception
{
public:
    explicit invalid_video_compression(const std::string& comp)
        : exception(fmt::format("Provided image compression '{}' is not supported. Only mp4 are supported", comp))
    {
    }
};

class invalid_segment_mask_compression : public exception
{
public:
    explicit invalid_segment_mask_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided 'segment_mask' compression '{}' is not supported. Only null, lz4 and png are supported", comp))
    {
    }
};

class invalid_medical_compression : public exception
{
public:
    explicit invalid_medical_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided 'medical' compression '{}' is not supported. Only nii, nii.gz and dcm are supported", comp))
    {
    }
};

class invalid_audio_compression : public exception
{
public:
    explicit invalid_audio_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided 'audio' compression '{}' is not supported. Only null, wav and mp3 are supported", comp))
    {
    }
};

class unexpected_medical_type_input_data : public exception
{
public:
    explicit unexpected_medical_type_input_data(std::string_view dtype)
        : exception(fmt::format(
              "MedicalType column expects compressed binary data as input, but received data with type '{}'", dtype))
    {
    }
};
class unexpected_input_data_for_dicom_column : public exception
{
public:
    explicit unexpected_input_data_for_dicom_column()
        : exception("MedicalType column with DICOM compression got unexpected NIFTI compressed data ad an input")
    {
    }
};

class unexpected_audio_type_input_data : public exception
{
public:
    explicit unexpected_audio_type_input_data(std::string_view dtype)
        : exception(fmt::format(
              "AudioType column expects compressed binary data as input, but received data with type '{}'", dtype))
    {
    }
};

class unexpected_input_data_for_audio_column : public exception
{
public:
    explicit unexpected_input_data_for_audio_column()
        : exception("AudioType column expects wav or mp3 compression, got unknown")
    {
    }
};

class invalid_binary_mask_compression : public exception
{
public:
    explicit invalid_binary_mask_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided 'binary_mask' compression '{}' is not supported. Only null and lz4 are supported", comp))
    {
    }
};

class unknown_bbox_type_pixel_format : public exception
{
public:
    explicit unknown_bbox_type_pixel_format(const std::string& c)
        : exception(fmt::format("The coordinate system pixel format '{}' is unknown", c))
    {
    }
};

class unknown_bbox_type : public exception
{
public:
    explicit unknown_bbox_type(const std::string& format)
        : exception(fmt::format("The coordinate format '{}' is unknown", format))
    {
    }
};

class invalid_polygon_shape : public exception
{
public:
    explicit invalid_polygon_shape(const icm::shape& sh)
        : exception(fmt::format("Input Polygon shape '{}' is invalid ", sh))
    {
    }
};

class invalid_link_data : public exception
{
public:
    explicit invalid_link_data(std::string_view dtype)
        : exception(fmt::format(
              "Data for the column with links must be of string type, but a different type was provided '{}'", dtype))
    {
    }
};

class embedding_size_mismatch : public exception
{
public:
    explicit embedding_size_mismatch(uint32_t exp, uint32_t got)
        : exception(fmt::format("Data for the embedding column must have {} size provided {}", exp, got))
    {
    }

    explicit embedding_size_mismatch()
        : exception(fmt::format("Data for the embedding column must have size, provided empty array"))
    {
    }
};

class wrong_video_dtype : public exception
{
public:
    explicit wrong_video_dtype(nd::dtype dtype)
        : exception(fmt::format("Video type only supports byte arrays, but got '{}'", nd::dtype_to_str(dtype)))
    {
    }
};

class invalid_mesh_compression : public exception
{
public:
    explicit invalid_mesh_compression(const std::string& comp)
        : exception(fmt::format(
              "Provided 'mesh' compression '{}' is not supported. Only stl and ply are supported", comp))
    {
    }
};

class unexpected_mesh_type_input_data : public exception
{
public:
    explicit unexpected_mesh_type_input_data(std::string_view dtype)
        : exception(fmt::format(
              "MeshType column expects compressed binary data as input, but received data with type '{}'", dtype))
    {
    }
};

class unexpected_input_data_for_mesh_column : public exception
{
public:
    explicit unexpected_input_data_for_mesh_column(const std::string& expected)
        : exception(fmt::format("MeshType column with {} compression got unexpected mesh format data as input", expected))
    {
    }
};

} // namespace deeplake_core
