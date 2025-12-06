#pragma once

#include <async/queue.hpp>
#include <base/logger_adapter.hpp>
#include <cstdint>
#include <memory>

namespace deeplake_api {

void initialize(std::shared_ptr<base::logger_adapter> logger_adapter, int32_t storage_concurrency = -1);

void deinitialize();

} // namespace deeplake_api
