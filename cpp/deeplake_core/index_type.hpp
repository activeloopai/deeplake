#pragma once

#include "basic_index_type.hpp"
#include "deeplake_index_type.hpp"
#include "embedding_index_type.hpp"

#include <icm/const_json.hpp>
#include <icm/json.hpp>

#include <variant>

namespace deeplake_core {

class type;

class index_type
{
public:
    using variant_type = std::variant<text_index_type,
                                      embedding_index_type,
                                      embeddings_matrix_index_type,
                                      json_index_type,
                                      numeric_index_type>;

public:
    explicit index_type(variant_type index)
        : type_(index)
    {
    }

    index_type(text_index_type text_index)
        : type_(text_index)
    {
    }

    index_type(embedding_index_type embedding_index)
        : type_(embedding_index)
    {
    }

    index_type(json_index_type json_index)
        : type_(json_index)
    {
    }

    index_type(numeric_index_type numeric_index)
        : type_(numeric_index)
    {
    }

    index_type(embeddings_matrix_index_type embeddings_matrix_index)
        : type_(embeddings_matrix_index)
    {
    }

    bool operator==(const index_type& other) const noexcept = default;

    static index_type from_json(const icm::const_json& json);

    icm::json to_json() const;

    std::string_view to_string() const;

    bool is_none() const noexcept
    {
        return std::holds_alternative<text_index_type>(type_) &&
               std::get<text_index_type>(type_) == deeplake_index_type::type::none;
    }

    bool is_text_index() const noexcept
    {
        return std::holds_alternative<text_index_type>(type_);
    }

    text_index_type as_text_index() const
    {
        return std::get<text_index_type>(type_);
    }

    bool is_embedding_index() const noexcept
    {
        return std::holds_alternative<embedding_index_type>(type_);
    }

    embedding_index_type as_embedding_index() const
    {
        return std::get<embedding_index_type>(type_);
    }

    bool is_embeddings_matrix_index() const noexcept
    {
        return std::holds_alternative<embeddings_matrix_index_type>(type_);
    }

    embeddings_matrix_index_type as_embeddings_matrix_index() const
    {
        return std::get<embeddings_matrix_index_type>(type_);
    }

    bool is_json_index() const noexcept
    {
        return std::holds_alternative<json_index_type>(type_);
    }

    json_index_type as_json_index() const
    {
        return std::get<json_index_type>(type_);
    }

    bool is_numeric_index() const noexcept
    {
        return std::holds_alternative<numeric_index_type>(type_);
    }

    numeric_index_type as_numeric_index() const
    {
        return std::get<numeric_index_type>(type_);
    }

    bool can_create_index_on_type(const type& t) const;

private:
    variant_type type_;
};

} // namespace deeplake_core
