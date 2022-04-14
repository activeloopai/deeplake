#include <torch/extension.h>
#include <vector>

#include <coroutine>


#include <concepts>
#include <exception>
#include <iostream>

#include <scheduler.h>
#include <requests.h>

#include <iostream>
// #include <boost/asio.hpp>
// #include <boost/thread/thread.hpp>
// #include <boost/bind/bind.hpp>




template <typename Found>
root_task Corofetch(Found on_found) {
  int a = 5;
  auto x = co_await prefetch(a);
  co_return on_found(x); // add to the stack
}



class PrefetchIterator{
private: 
  int a;
  std::vector<int> v;
  py::object ref; // keep a reference
  size_t index = 0;

public:
  PrefetchIterator(){
    //myCoroutineResult = myCoroutineFunction();
    v = {1, 2, 3};
    // boost::asio::io_context io;
    // boost::asio::steady_timer t(io, boost::asio::chrono::seconds(5));
    
    //boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
    // io.run();
    int streams = 1000;
    int calls = 10000;
    size_t found_count = 0;

    throttler t(streams);
    
    std::cout << found_count << std::endl;
    for (int i = 1; i < calls; i++)
      t.spawn(Corofetch([&](auto) { ++found_count; }));
    t.run();

    std::cout << "Prefetch iterator started!" << std::endl;
    std::cout << found_count << std::endl;
  }

  int next(){
    if (index == v.size())
      throw py::stop_iteration(); // Great
    return v[index++]; //(myCoroutineResult.next());
  }

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_request", &simple_request, "simple request");
  pybind11::class_<PrefetchIterator>(m, "prefetch")
            .def(pybind11::init())
            .def("__next__", &PrefetchIterator::next)
            .def("__iter__", [](PrefetchIterator &it) -> PrefetchIterator& { return it; }, py::keep_alive<0, 1>());
}
