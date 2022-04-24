#include "iostream"

#include "eventuals/eventual.h"
#include "eventuals/http.h"
#include "eventuals/foreach.h"
#include "eventuals/closure.h"
#include "eventuals/range.h"
#include "eventuals/terminal.h"
#include "eventuals/then.h"
#include "eventuals/http.h"

#include "eventuals/interrupt.h"
#include "eventuals/let.h"
#include "eventuals/scheduler.h"
#include "eventuals/type-traits.h"


int main() {
  eventuals::EventLoop::ConstructDefault();


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
  std::cout << __LINE__ << std::endl;
  auto [future, k] = eventuals::Terminate(std::move(e()));
  k.Start();
  std::cout << __LINE__ << std::endl;

  eventuals::EventLoop::Default().RunUntil(future);

  future.get();
  std::cout << "done" << std::endl;
  eventuals::EventLoop::DestructDefault();
}
