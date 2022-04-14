#include <torch/extension.h>
#include <vector>

#include <coroutine>


#include <concepts>
#include <exception>
#include <iostream>

#include <scheduler.h>
#include <requests.h>

#include <iostream>


template <typename Found>
root_task Corofetch(int a, Found on_found) {
  auto x = co_await prefetch(a);
  //std::cout << x << std::endl;
  co_return on_found(x); // add to the stack
}


class PrefetchIterator{
private: 
  int a;
  std::vector<int> v;
  py::object ref; // keep a reference
  size_t index = 0;
  std::list<std::string> answer;

public:
  PrefetchIterator(){
    //myCoroutineResult = myCoroutineFunction();

    int requests_on_the_fly = 10;
    int calls = 100;
    size_t found_count = 0;
    answer = std::list<std::string>();

    throttler t(requests_on_the_fly);
    std::cout << found_count << std::endl;
    for (int i = 1; i < calls; i++)
      t.spawn(Corofetch(i, [&](auto x) {answer.push_back(x); found_count++; }));
    t.run();

    std::cout << "Prefetch iterator started!" << std::endl;
    std::cout << found_count << std::endl;
  }

  auto next(){
    if (index == answer.size())
      throw py::stop_iteration();
    auto var = answer.front();
    answer.pop_front();
    return py::bytes(var); //(myCoroutineResult.next());
  }

};

/* 
TODO
1. [todo] coroutine returns reference to pybytes
2. [todo] coroutines run async while iterator is running
3. [todo] pybytes returned to iterator
4. [todo] AWS requests are sent
5. [tests] if requests are really async 
*/ 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_request", &simple_request, "simple request");
  pybind11::class_<PrefetchIterator>(m, "prefetch")
            .def(pybind11::init())
            .def("__next__", &PrefetchIterator::next)
            .def("__iter__", [](PrefetchIterator &it) -> PrefetchIterator& { return it; }, py::keep_alive<0, 1>());
}
