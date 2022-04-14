#include <torch/extension.h>
#include <vector>

#include <scheduler.h>
#include <requests.h>


template <typename Found>
root_task Corofetch(int a, Found on_found) {
  auto x = co_await prefetch(a);
  co_return on_found(x);
}


class PrefetchIterator{
private: 
  py::object ref; // keep a reference
  std::list<std::string> answer;
  throttler t = throttler(10);

public:
  PrefetchIterator(int requests_on_the_fly = 10, int calls = 10){

    answer = std::list<std::string>();
    for (int i = 1; i < calls; i++)
      t.reg(Corofetch(i, [&](auto x) {answer.push_back(x);}));
      
  }

  auto next(){
    t.next();    
    if (answer.size() == 0)
      throw py::stop_iteration();
    auto var = answer.front();
    answer.pop_front();
    return py::bytes(var);
  }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_request", &simple_request, "simple request");
  pybind11::class_<PrefetchIterator>(m, "prefetch")
            .def(pybind11::init())
            .def("__next__", &PrefetchIterator::next)
            .def("__iter__", [](PrefetchIterator &it) -> PrefetchIterator& { return it; }, py::keep_alive<0, 1>());
}
