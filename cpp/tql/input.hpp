#pragma once

#include "exceptions.hpp"
#include "function_variant.hpp"
#include "functions_registry.hpp"
#include "table.hpp"
#include "tensor_property.hpp"

#include <async/async.hpp>
#include <async/promise.hpp>
#include <heimdall/column_view.hpp>
#include <heimdall/dataset_view.hpp>
#include <heimdall_common/chained_column_view.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tql {

struct input_tensor
{
    input_tensor(std::string ds_alias, std::string name, tensor_property p)
        : ds_alias(std::move(ds_alias)), name(std::move(name))
    {
        properties.insert(p);
    }

    auto operator<=>(const input_tensor&) const = default;

    std::string ds_alias;
    std::string name;
    std::set<tensor_property> properties;
};

struct input_functor
{
    input_functor(std::string name, int num_args)
        : name(std::move(name))
        , num_args(num_args)
    {
    }

    std::string name;
    int num_args;
};
inline bool operator==(const input_functor& lhs, const input_functor& rhs)
{
    return lhs.name == rhs.name && lhs.num_args == rhs.num_args;
}

struct input
{
    void add_tensor(std::string ds_alias, std::string name, tql::tensor_property p)
    {
        auto i = std::ranges::find_if(tensors, [&ds_alias, &name](const auto& t) {
            return t.ds_alias == ds_alias && t.name == name;
        });
        if (i != tensors.end()) {
            i->properties.insert(p);
        } else {
            tensors.emplace_back(std::move(ds_alias), std::move(name), p);
        }
    }

    void add_data(std::string ds_alias, std::string column_name, int64_t index)
    {
        auto i = std::ranges::find(data, query_core::data_reference{ds_alias, column_name, index});
        if (i == data.end()) {
            data.emplace_back(std::move(ds_alias), std::move(column_name), index);
        }
    }

    void add_function(const std::string& n, int a)
    {
        auto i = std::ranges::find(functors, input_functor{n, a});
        if (i == functors.end()) {
            functors.emplace_back(n, a);
        }
    }

    std::vector<input_tensor> tensors;
    std::vector<query_core::data_reference> data;
    std::vector<input_functor> functors;
};

inline async::promise<query_core::static_data_t> request_data(const std::vector<query_core::data_reference>& data,
                                                                 table& dataset)
{
    std::vector<async::promise<nd::array>> requests;
    for (const auto& [ds_alias, column_name, index] : data) {
        auto& t = dataset.find_tensor(ds_alias, column_name);
        requests.push_back(t.request_sample(index, storage::fetch_options()));
    }
    return async::combine(std::move(requests)).then([data](auto samples) {
        query_core::static_data_t res;
        ASSERT(samples.size() == data.size());
        for (auto i = 0; i < data.size(); ++i) {
            res.try_emplace(data[i], samples[i]);
        }
        return res;
    });
}

using functions_t = std::map<std::string, function_variant, std::less<>>;

inline async::promise<functions_t> request_functions(const std::vector<input_functor>& functions, table& dataset)
{
    auto function_registry_finder = [](heimdall::column_view& t) {
        auto r = dynamic_cast<functions_registry*>(&t);
        if (r == nullptr) {
            auto& orig = heimdall_common::original_tensor_over_chain(t);
            r = dynamic_cast<functions_registry*>(&orig);
        }
        return r;
    };
    std::vector<async::promise<function_variant>> requests;
    for (const auto& [name, args_count] : functions) {
        bool found = false;
        for (auto& [_, ds] : dataset.sub_datasets()) {
            for (auto& t : *ds) {
                auto r = function_registry_finder(t);
                if (r == nullptr) {
                    continue;
                }
                auto fs = r->functions_names();
                auto it = std::ranges::find(fs, name);
                if (it == fs.end()) {
                    continue;
                }
                requests.push_back(r->get_function(name, args_count));
                found = true;
                break;
            }
        }
        if (!found) {
            throw external_function_not_found(name);
        }
    }
    return async::combine(std::move(requests)).then([functions](auto fs) {
        functions_t res;
        ASSERT(fs.size() == functions.size());
        for (auto i = 0; i < functions.size(); ++i) {
            res.try_emplace(functions[i].name, fs[i]);
        }
        return res;
    });
}

} // namespace tql
