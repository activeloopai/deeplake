
#pragma once

#include "progress_utils.hpp"

#include <deeplake_api/dataset.hpp>
#include <icm/string_map.hpp>

#include <memory>

namespace pg::utils {

inline void append_row(std::shared_ptr<deeplake_api::dataset> dataset, icm::string_map<nd::array> row)
{
    async::run_on_main([dataset = std::move(dataset), row = std::move(row)](){
        return dataset->append_row(std::move(row));
    }).get_future().get();
}

inline void commit_dataset(std::shared_ptr<deeplake_api::dataset> dataset, bool show_progress)
{
    constexpr auto high_num_rows = 50000;
    const bool print_progress = (show_progress && dataset->num_rows() > high_num_rows && dataset->has_uncommitted_changes());
    auto promise = async::run_on_main([ds = std::move(dataset)](){ return ds->commit(); });
    if (print_progress) {
        const std::string message = fmt::format("Committing dataset (samples: {})", dataset->num_rows());
        pg::utils::print_progress_and_wait(std::move(promise), message);
    } else {
        promise.get_future().get();
    }
}

} // namespace pg::utils