#pragma once

#include "chained_dataset_view.hpp"

#include <heimdall/dataset_view.hpp>

namespace heimdall_common {

template <typename T>
std::vector<std::shared_ptr<T>> original_dataset_objects_over_chain(heimdall::dataset_view_ptr d)
{
    std::vector<std::shared_ptr<T>> original_views;
    auto push_back_original_views = [&original_views](heimdall::dataset_view_ptr d) {
        while (auto q = dynamic_cast<chained_dataset_view*>(d.get())) {
            d = q->source();
        }
        auto o = std::dynamic_pointer_cast<T>(d);
        if (o) {
            original_views.push_back(o);
        }
    };
    std::function<void(heimdall::dataset_view_ptr)> get_original_views = [&](heimdall::dataset_view_ptr d) {
        if (auto ds = dynamic_cast<multiple_chained_dataset*>(d.get())) {
            for (const auto item : ds->sources()) {
                get_original_views(item);
            }
        } else {
            push_back_original_views(d);
        }
    };
    get_original_views(d);
    return original_views;

}

}
