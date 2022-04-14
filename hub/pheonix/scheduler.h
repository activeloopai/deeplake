#pragma once

#include <xmmintrin.h>
#include <coroutine>
#include <stdint.h>
#include <cstdio>

/* Almost Gor Nishanov's code */

///// --- INFRASTRUCTURE CODE BEGIN ---- ////

struct scheduler_queue {
  static constexpr const int N = 256;
  using coro_handle = std::coroutine_handle<>;

  uint32_t head = 0;
  uint32_t tail = 0;
  coro_handle arr[N];

  void push_back(coro_handle h) {
    arr[head] = h;
    head = (head + 1) % N;
  }

  coro_handle pop_front() {
    auto result = arr[tail];
    tail = (tail + 1) % N;
    return result;
  }
  auto try_pop_front() { return head != tail ? pop_front() : coro_handle{}; }

  void run() {
    while (auto h = try_pop_front())
      h.resume();
  }
};

inline scheduler_queue scheduler;

// Simple thread caching allocator.
struct tcalloc {
  struct header {
    header *next;
    size_t size;
  };
  header *root = nullptr;
  size_t last_size_allocated = 0;
  size_t total = 0;
  size_t alloc_count = 0;

  ~tcalloc() {
    auto current = root;
    while (current) {
      auto next = current->next;
      ::free(current);
      current = next;
    }
  }

  void *alloc(size_t sz) {
    if (root && root->size <= sz) {
      void *mem = root;
      root = root->next;
      return mem;
    }
    ++alloc_count;
    total += sz;
    last_size_allocated = sz;

    return malloc(sz);
  }

  void stats() {
    printf("allocs %zu total %zu sz %zu\n", alloc_count, total, last_size_allocated);
  }

  void free(void *p, size_t sz) {
    auto new_entry = static_cast<header *>(p);
    new_entry->size = sz;
    new_entry->next = root;
    root = new_entry;
  }
};

inline tcalloc allocator;


struct throttler;

struct root_task {
  struct promise_type;
  using HDL = std::coroutine_handle<promise_type>;

  struct promise_type {
    throttler *owner = nullptr;

    void *operator new(size_t sz) { return allocator.alloc(sz); }
    void operator delete(void *p, size_t sz) { allocator.free(p, sz); }

    root_task get_return_object() { return root_task{*this}; }
    std::suspend_always initial_suspend() { return {}; }
    void return_void();
    void unhandled_exception() noexcept { std::terminate(); }
    std::suspend_never final_suspend() noexcept(true) { return {}; }
  };

  // TODO: this can be done via a wrapper coroutine
  auto set_owner(throttler *owner) {
    auto result = h;
    h.promise().owner = owner;
    h = nullptr;
    return result;
  }

  ~root_task() {
    if (h)
      h.destroy();
  }

  root_task(root_task&& rhs) : h(rhs.h) { rhs.h = nullptr; }
  root_task(root_task const&) = delete;

private:
  root_task(promise_type &p) : h(HDL::from_promise(p)) {}

  HDL h;
};

struct throttler {
  unsigned limit;
  unsigned n_tasks = 0;
  explicit throttler(unsigned limit) : limit(limit) {}

  void on_task_done() { ++limit; --n_tasks;}

  void spawn(root_task t) {
    if (limit == 0)
      scheduler.pop_front().resume();

    auto h = t.set_owner(this);
    scheduler.push_back(h);
    --limit;
  }

  void run() {
    scheduler.run();
  }

  void reg(root_task t) {
    auto h = t.set_owner(this);
    scheduler.push_back(h);
    n_tasks++;
  }

  void next() {
    if (n_tasks == 0){
      return;
    }
    scheduler.pop_front().resume();
  }

  ~throttler() { run(); }
};

void root_task::promise_type::return_void() { owner->on_task_done(); }

///// --- INFRASTRUCTURE CODE END ---- ////