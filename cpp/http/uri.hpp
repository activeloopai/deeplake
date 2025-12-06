#pragma once

#include "exceptions.hpp"
#include "url.hpp"

#include <regex>
#include <string>

/**
 * @brief common uri interface for different type uri handling
 *
 */

namespace http {

class uri
{
public:
    explicit uri(std::string path)
        : uri_(normalize_url(path))
    {
        if (auto branch_pos = uri_.find('@'); branch_pos != std::string::npos) {
            branch_ = uri_.substr(branch_pos + 1);
            uri_ = uri_.substr(0, branch_pos);
        }
        if (is_gcs_path() || is_http_path()) {
            if (uri_.ends_with('/')) {
                uri_.pop_back();
            }
        } else {
            if (!uri_.ends_with('/')) {
                uri_.push_back('/');
            }
        }
    }

    void verify() const
    {
        if (is_s3_path() || is_gcs_path() || is_azure_path()) {
            std::regex uri_regex(R"(^([a-z0-9]+)://([^/]+)/(.+)$)");
            std::smatch matches;

            if (!std::regex_match(uri_, matches, uri_regex)) {
                throw invalid_uri(uri_);
            }
        }
    }

    bool is_gcs_path() const noexcept
    {
        return uri_.starts_with("gs://") || uri_.starts_with("gcs://") || uri_.starts_with("gcp://");
    }

    bool is_s3_path() const noexcept
    {
        return uri_.starts_with("s3://");
    }

    bool is_http_path() const noexcept
    {
        return uri_.starts_with("http://") || uri_.starts_with("https://");
    }

    bool is_azure_path() const noexcept
    {
        return uri_.starts_with("azure://") || uri_.starts_with("az://");
    }

    bool is_filesystem_path() const noexcept
    {
        return uri_.starts_with("file://") || uri_.find("://") == std::string::npos;
    }

    bool is_hub_path() const noexcept
    {
        return uri_.starts_with("al://") || uri_.starts_with("hub://");
    }

    bool is_mem_path() const noexcept
    {
        return uri_.starts_with("mem://");
    }

    bool is_tmp_path() const noexcept
    {
        return uri_.starts_with("tmp://");
    }

    std::string path() const noexcept;

    std::string_view protocol() const noexcept;

    std::string_view hostname() const noexcept;

    std::string_view subpath() const noexcept;

    std::string url() const noexcept
    {
        if (branch_.empty()) {
            return uri_;
        }
        return uri_ + "@" + branch_;
    }

    const std::string& branch() const noexcept
    {
        return branch_;
    }

private:
    std::string uri_;
    std::string branch_;
};

} // namespace http
