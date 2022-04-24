#include "iostream"

#include "eventuals/closure.h"
#include "eventuals/eventual.h"
#include "eventuals/foreach.h"
#include "eventuals/http.h"
#include "eventuals/range.h"
#include "eventuals/reduce.h"
#include "eventuals/terminal.h"
#include "eventuals/then.h"

#include "eventuals/conditional.h"
#include "eventuals/interrupt.h"
#include "eventuals/let.h"
#include "eventuals/map.h"
#include "eventuals/scheduler.h"
#include "eventuals/stream.h"
#include "eventuals/type-traits.h"

// #include "eventuals/if.h" // No such file or directory

int main() {
  eventuals::EventLoop::ConstructDefault();

  auto request = [](auto i) {
    std::cout << i << std::endl;
    return eventuals::http::Get("http://localhost:8000/") |
           eventuals::Then([&](auto response) { return response.code; });
  };

  auto e = [&]() {
    return eventuals::Stream<int>()
               .context(2)
               .next([](auto &count, auto &k) {
                 if (count > 0) {
                   k.Emit(count--);
                 } else {
                   k.Ended();
                 }
               })
               .done([](auto &, auto &k) { k.Ended(); }) |
           eventuals::Map([&](int i) {
             std::cout << i << std::endl;
             std::cout << "request" << std::endl;
             return request(i); // fails on second request
           }) |
           eventuals::Reduce(
               /* sum = */ 0, [](auto &sum) {
                 return eventuals::Then([&](auto &&value) {
                   // auto resp = value();
                   std::cout << "get first value" << std::endl;
                   sum += value;
                   std::cout << sum << std::endl;
                   return true;
                 });
               });
  };

  /*
  std::cout << __LINE__ << std::endl;
  auto e = []() {
    return eventuals::Closure([v = std::vector<int>()]() mutable {
      std::string url = "activeloop.ai";
      return eventuals::Foreach(
                 eventuals::Range(5),
                 [&](int i) {
                   v.push_back(i);
                 })
          | eventuals::http::Get(url)
          | eventuals::Then([&](auto x) {
            std::cout << x.code << std::endl; // v.push_back(x.code);
            return x.code;
           })
          | eventuals::Then([&](auto x) {
               std::cout << __LINE__ << std::endl;
               return "activeloop.ai"; //std::move(v);
             });
    });
  };
  */

  std::cout << __LINE__ << std::endl;
  auto [future, k] = eventuals::Terminate(std::move(e()));
  k.Start();
  std::cout << __LINE__ << std::endl;

  eventuals::EventLoop::Default().RunUntil(future);

  future.get();
  std::cout << "done" << std::endl;
  eventuals::EventLoop::DestructDefault();
}
