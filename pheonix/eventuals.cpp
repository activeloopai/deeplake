#include <iostream>

#include "eventuals/closure.h"
#include "eventuals/collect.h"
#include "eventuals/concurrent.h"
#include "eventuals/conditional.h"
#include "eventuals/eventual.h"
#include "eventuals/foreach.h"
#include "eventuals/http.h"
#include "eventuals/if.h"
#include "eventuals/interrupt.h"
#include "eventuals/let.h"
#include "eventuals/map.h"
#include "eventuals/range.h"
#include "eventuals/reduce.h"
#include "eventuals/scheduler.h"
#include "eventuals/stream.h"
#include "eventuals/terminal.h"
#include "eventuals/then.h"
#include "eventuals/timer.h"
#include "eventuals/type-traits.h"

int main() {
  eventuals::EventLoop::ConstructDefaultAndRunForeverDetached();

  auto request = [](auto i) {
    std::cout << i << std::endl;
    return eventuals::Timer(std::chrono::seconds(1))
        | eventuals::Then([]() {
             return eventuals::http::Get("http://www.google.com")
                 | eventuals::Then([&](auto response) {
                      return response.code();
                    });
           });
  };

  auto e = [&]() {
    return eventuals::Range(2)
        | eventuals::Concurrent([&]() {
             return eventuals::Map([&](int i) {
               return request(i);
             });
           })
        | eventuals::Collect<std::vector<int>>();
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
  auto [future, k] = eventuals::Terminate(e());
  k.Start();
  std::cout << __LINE__ << std::endl;

  std::vector<int> codes = future.get();

  for (int code : codes) {
    std::cout << "Received HTTP " << code << std::endl;
  }

  std::cout << "done" << std::endl;
}
