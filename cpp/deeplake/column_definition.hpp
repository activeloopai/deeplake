#pragma once

#include <async/promise.hpp>
#include <heimdall/column_definition_view.hpp>

#include <string>

namespace deeplake {

namespace impl {
class data_container_base;
}

class column_definition
{
public:
    column_definition(heimdall::column_id_t id,
                      std::string name,
                      deeplake_core::type type,
                      impl::data_container_base* container)
        : column_definition_view_(heimdall::column_definition_view(id, name, type))
        , container_(container)
    {
    }

    column_definition(const heimdall::column_definition_view& column_view, impl::data_container_base* container)
        : column_definition_view_({column_view})
        , container_(container)
    {
    }

    [[nodiscard]] heimdall::column_id_t id() const
    {
        return column_definition_view_.id();
    }

    [[nodiscard]] const std::string& name() const
    {
        return column_definition_view_.name();
    }

    [[nodiscard]] const deeplake_core::type& core_type() const
    {
        return column_definition_view_.core_type();
    }

    [[nodiscard]] std::string to_string() const
    {
        return column_definition_view_.to_string();
    }

    [[nodiscard]] async::promise<void> rename(const std::string& new_name);

    [[nodiscard]] async::promise<void> drop();

private:
    heimdall::column_definition_view column_definition_view_;
    impl::data_container_base* container_;
};

} // namespace deeplake
