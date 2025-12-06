#pragma once

#include <heimdall/schema_view.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace deeplake {

namespace impl {
class data_container_base;
} // namespace impl

class column_definition;

class schema
{
    friend impl::data_container_base;

public:
    explicit schema(impl::data_container_base* container)
        : container_(container)
        , schema_view_(std::make_shared<heimdall::schema_view>())
    {
    }

    [[nodiscard]] std::vector<column_definition> columns() const;

    [[nodiscard]] column_definition get_column(const std::string& name) const;

    [[nodiscard]] heimdall::column_id_t get_column_id(const std::string& name) const;

    [[nodiscard]] const std::vector<heimdall::column_definition_view>& columns_view() const
    {
        return schema_view_->columns_view();
    }

    [[nodiscard]] int size() const
    {
        return schema_view_->size();
    }

    [[nodiscard]] heimdall::column_definition_view get_column_view(const std::string& name) const
    {
        return schema_view_->get_column_view(name);
    }

    [[nodiscard]] heimdall::column_definition_view get_column_view(heimdall::column_id_t column_id) const
    {
        return schema_view_->get_column_view(column_id);
    }

    [[nodiscard]] std::string to_string() const
    {
        return schema_view_->to_string();
    }

    [[nodiscard]] nd::schema core_schema() const
    {
        return schema_view_->core_schema();
    }

private:
    void add_column(heimdall::column_id_t id,
                    const std::string& name,
                    const deeplake_core::type& type,
                    const std::optional<nd::array>& default_value = std::nullopt);

    void rename(heimdall::column_id_t id, const std::string& name);

    void drop_column(heimdall::column_id_t id);

    void resize_embedding_column(heimdall::column_id_t id, int32_t size);

    void clear();

private:
    impl::data_container_base* container_;
    std::shared_ptr<heimdall::schema_view> schema_view_;
};

} // namespace deeplake
