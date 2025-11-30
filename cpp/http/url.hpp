#pragma once

#include <algorithm>
#include <regex>
#include <string>

namespace http {

inline bool is_gcs_path(std::string_view url)
{
    return url.starts_with("gs://") || url.starts_with("gcs://") || url.starts_with("gcp://");
}

inline bool is_s3_path(std::string_view url)
{
    return url.starts_with("s3://");
}

inline bool is_http_path(std::string_view url)
{
    return url.starts_with("http://") || url.starts_with("https://");
}

inline bool is_azure_path(std::string_view url)
{
    return url.starts_with("azure://") || url.starts_with("az://");
}

inline bool is_filesystem_path(std::string_view url)
{
    return url.starts_with("file://") || url.find("://") == std::string::npos;
}

inline std::string hostname_from_url(const std::string& url)
{
    auto p = url.find('/', 8);
    if (p == std::string::npos) {
        return url;
    }
    return url.substr(0, p);
}

inline std::string normalize_url(const std::string& url)
{
    // remove all double slashes after https:// and before ?
    std::string result = url;
    std::string::size_type pos = result.find("://");
    if (pos != std::string::npos) {
        pos += 3; // Skip past "://"
        std::string::size_type end = result.find('?', pos);
        if (end == std::string::npos) {
            end = result.length();
        }
        std::string path = result.substr(pos, end - pos);
        path = std::regex_replace(path, std::regex(R"(//+)"), "/");
        result = result.substr(0, pos) + path + result.substr(end);
    }
    return result;
}

} // namespace http
