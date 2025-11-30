#pragma once

/**
 * @file queue.hpp
 * @brief Definition of the `queue` class.
 */
#include "priority.hpp"

#include <base/function.hpp>

#include <atomic>
#include <memory>

namespace async {

/**
 * @brief Priority queue to submit and execute asynchronous tasks.
 * The queue is used to submit tasks to be executed asynchronously.
 * The tasks are executed in the order of their priority.
 * The tasks can be submitted with the priority, which can be changed later. The tasks can be cancelled or waited to
 * finish.
 * The queue can be paused and resumed. When the queue is paused, the tasks are not executed, but they are not
 * removed.
 */
class queue
{
protected:
    class impl;
    std::shared_ptr<impl> impl_;

protected:
    explicit queue(std::shared_ptr<impl> i)
        : impl_(std::move(i))
    {
    }

    queue(const queue&) = delete;
    queue& operator=(const queue&) = delete;
    queue(queue&&) = delete;
    queue& operator=(queue&&) = delete;
    ~queue();

    /// @name Type definitions
    /// @{
public:
    /**
     * @brief Class representing unique id of the task submitted to the queue.
     */
    class id_type
    {
        friend class impl;

    public:
        /**
         * @brief Default constructor.
         * The id is being default constructed and then passed to the `queue::submit` function as an argument. The
         * `submit` stores the id of the task.
         */
        id_type() = default;

        id_type(const id_type&) = delete;
        id_type& operator=(const id_type&) = delete;
        id_type(id_type&&) = delete;
        id_type& operator=(id_type&&) = delete;

        ~id_type() noexcept;

        /**
         * @brief Returns `true` if the task is still valid. If the task is completed, this conversion will return
         * `false`.
         */
        explicit inline operator bool() const noexcept
        {
            return (queue_impl_.load(std::memory_order_acquire) != nullptr);
        }

        /// Priority of the task.
        int priority() const;

        /// Change the priority of the task.
        void set_priority(int p) const;

        /**
         * @brief Cancel the task.
         * If the task is already running, it will not be cancelled. To wait for the task to finish please use
         * `remove_or_wait` function.
         */
        void remove() const noexcept;

        /**
         * @brief Cancel the task or waits to finish.
         * If the task is already running, this function will wait for its completion.
         */
        void remove_or_wait() const;

        /// Waits for the task to finish.
        void wait() const noexcept;

        /// Reset the id connection to the queue.
        void reset() noexcept;

    private:
        std::atomic<impl*> queue_impl_ = nullptr;
        std::atomic<int> worker_id_ = -1;
        int task_id_ = -1;
    };
    /// @}

public:
    /**
     * @brief This function will handle exceptions thrown by the tasks.
     * @note The functor should be thread safe.
     */
    void set_exception_handler(base::function<void(std::exception_ptr)> handler) noexcept;

    /// Checks if the calling thread is a worker thread of the queue.
    bool is_this_thread_worker() const noexcept;

public:
    /**
     * @brief Pauses the queue execution.
     * If the queue is paused, this function has no any effect.
     * The execution can be resumed by `resume` function.
     */
    void pause() noexcept;

    /**
     * @brief Resumes the queue execution.
     * If the queue is not paused, this function has no any effect.
     */
    void resume() noexcept;

public:
    /**
     * @brief Submits the given function to the queue with the highest priority.
     * @param f Function to run in queue.
     * @param task_id If non-null, then the id of the task will be stored in `task_id`.
     */
    inline void submit(base::function<void()>&& f, id_type* task_id = nullptr)
    {
        submit(std::move(f), max_priority, task_id);
    }

    /**
     * @brief Submits the given function to the queue with the given priority.
     * @param f Function to run in queue.
     * @param task_id If non-null, then the id of the task will be stored in `task_id`.
     */
    void submit(base::function<void()>&& f, int priority, id_type* task_id = nullptr);

    /**
     * @brief Runs the given function in the current thread if it is a worker thread of the queue; otherwise, submits
     * the function
     * @param f Function to run in queue.
     * @param priority Priority of the task.
     * @param task_id If non-null, then the id of the task will be stored in `task_id`.
     */
    inline void run_or_submit(base::function<void()>&& f, int priority = max_priority, id_type* task_id = nullptr)
    {
        if (is_this_thread_worker()) {
            f();
        } else {
            submit(std::move(f), priority, task_id);
        }
    }

    /**
     * @brief Removes all tasks from the queue.
     */
    void clear() noexcept;

    /// Returns the number of task in the queue.
    unsigned size() const noexcept;

    /// Checks if the queue is empty.
    inline bool empty() const noexcept
    {
        return size() == 0;
    }

    /// Returns number of threads of this queue.
    int num_threads() const noexcept;
};


class main_queue final : public queue
{
public:
    /// Depends on the platform this may be initialized with a single worker thread.
    main_queue();

    /**
     * @brief Enable/disable detailed tracking (impacts performance)
     * @param enable Whether to track detailed metrics per task
     */
    void enable_detailed_pressure_tracking(bool enable);

    /// @name Internal functions not intended for public use.
    /// @{
public:
    static bool iterate_all();
    bool try_iterate();
    /// @}

private:
    class impl;
};

class bg_queue final : public queue
{
public:
    /**
     * @brief Creates queue with the given number of threads.
     *
     * Checks system available memory periodically if memory monitoring is enabled.
     * In case if there is no available memory the worker threads will sleep for some
     * time and again check for free memory to resume.
     *
     * @param num_threads Number of queue threads.
     * @param monitor_system_memory Enable system memory monitoring if true.
     */
    explicit bg_queue(int num_threads, bool monitor_system_memory = false);

    /**
     * @brief Changes the number of threads of the queue.
     * @param nt New number of threads.
     */
    void change_num_threads(int nt);

private:
    class impl;
};

} // namespace async

