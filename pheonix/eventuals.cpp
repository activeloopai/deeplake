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
    return eventuals::Range(10)
        | eventuals::Concurrent([&]() {
             return eventuals::Map([&](int i) {
               return request(i);
             });
           })
        | eventuals::Collect<std::vector<int>>();
  };

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
