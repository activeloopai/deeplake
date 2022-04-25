#include <pybind11/pybind11.h>
#include <vector>
#include "eventuals/callback.h"
#include "requests.h"
#include "scheduler.h"

namespace py = pybind11;

template <typename Found>
root_task Corofetch(std::string url, Found on_found) {
  auto resp = co_await prefetch(url);
  co_return on_found(resp);
}

class PrefetchIterator {
 private:
  py::object ref;  // keep a reference
  std::list<std::string> answer;
  throttler t = throttler(10);
  int counter;

 public:
  PrefetchIterator(py::list inlist) {
    // TOOD
    // 1. resolve the asyncronous request
    // 2. fix number of requests on the fly
    // 3. implement aws requests

    answer = std::list<std::string>();

    for (auto item : inlist) {
      t.reg(Corofetch(item.cast<std::string>(), [&](auto x) {
        answer.push_back(x);
        counter++;
      }));
    }
  }

  auto next() {
    std::cout << counter << std::endl;
    t.next();
    if (answer.size() == 0) throw py::stop_iteration();
    auto var = answer.front();
    answer.pop_front();
    return py::bytes(var);
  }
};

PYBIND11_MODULE(pheonix, m) {
  pybind11::class_<PrefetchIterator>(m, "prefetch")
      .def(pybind11::init<py::list &>())
      .def("__next__", &PrefetchIterator::next)
      .def("__iter__",
           [](PrefetchIterator &it) -> PrefetchIterator & { return it; },
           py::keep_alive<0, 1>());
}
