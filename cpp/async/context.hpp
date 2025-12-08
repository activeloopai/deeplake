#pragma once

#include <atomic>

namespace async {

namespace impl {
template <typename T>
class bg_queue_promise;
}

class context
{
public:
    std::atomic<float> progress = 0.f;
    inline bool is_cancelled() const
    {
        return is_cancelled_;
    }

private:
    std::atomic<bool> is_cancelled_ = false;

    template <typename T>
    friend class impl::bg_queue_promise;
};

} // namespace async