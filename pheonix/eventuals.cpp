
#include "eventuals/foreach.h"
#include "eventuals/closure.h"
#include "eventuals/range.h"
#include "eventuals/terminal.h"
#include "eventuals/then.h"
#include <iostream>

int main() {
  auto e = []() {
    return eventuals::Closure([v = std::vector<int>()]() mutable {
      return eventuals::Foreach(
                 eventuals::Range(5),
                 [&](int i) {
                   v.push_back(i);
                 })
          | eventuals::Then([&]() {
               return std::move(v);
             });
    });
  };

  e();
  std::cout << "done" << std::endl;
}