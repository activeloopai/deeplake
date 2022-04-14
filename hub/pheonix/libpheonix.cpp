#include <torch/extension.h>
#include <vector>

#include <coroutine>


#include <concepts>
#include <exception>
#include <iostream>

#include <scheduler.h>
#include <requests.h>

#include <iostream>

#include <chrono>
#include <thread>

template <typename Found>
root_task Corofetch(int a, Found on_found) {
  auto x = co_await prefetch(a);
  co_return on_found(x);
}


class PrefetchIterator{
private: 
  int a;
  std::vector<int> v;
  py::object ref; // keep a reference
  size_t index = 0;
  std::list<std::string> answer;
  throttler t = throttler(10);
  size_t found_count = 0;

public:
  PrefetchIterator(int requests_on_the_fly = 10, int calls = 10){
    //myCoroutineResult = myCoroutineFunction();

    answer = std::list<std::string>();

    for (int i = 1; i < calls; i++)
      t.reg(Corofetch(i, [&](auto x) {answer.push_back(x); found_count++; }));
    
    std::cout << "Prefetch iterator started!" << std::endl;
    std::cout << found_count << std::endl;
  }

  auto next(){
    t.next();    
    if (index == answer.size())
      throw py::stop_iteration();
    auto var = answer.front();
    answer.pop_front();
    return py::bytes(var); //(myCoroutineResult.next());
  }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_request", &simple_request, "simple request");
  pybind11::class_<PrefetchIterator>(m, "prefetch")
            .def(pybind11::init())
            .def("__next__", &PrefetchIterator::next)
            .def("__iter__", [](PrefetchIterator &it) -> PrefetchIterator& { return it; }, py::keep_alive<0, 1>());
}
