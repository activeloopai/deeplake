#include <torch/extension.h>

#include <iostream>
#include <vector>

#include <coroutine>
#include <curl/curl.h>


#include <concepts>
#include <coroutine>
#include <exception>
#include <iostream>


struct ReturnObject {
  struct promise_type {
    ReturnObject get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
  };
};

struct Awaiter {
  std::coroutine_handle<> *hp_;
  constexpr bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) { *hp_ = h; }
  constexpr void await_resume() const noexcept {}
};

ReturnObject
counter(std::coroutine_handle<> *continuation_out)
{
  Awaiter a{continuation_out};
  for (unsigned i = 0;; ++i) {
    co_await a;
    std::cout << "counter: " << i << std::endl;
  }
}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

auto simple_request(int a) {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;

  std::coroutine_handle<> h;
  counter(&h);
  for (int i = 0; i < 3; ++i) {
    std::cout << "In main1 function\n";
    h();
  }
  h.destroy();


  curl = curl_easy_init();
  if (curl){
    curl_easy_setopt(curl, CURLOPT_URL, "http://www.google.com");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
  }

  return py::bytes(readBuffer);
}




// try hunter gate or use cmake list
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_request", &simple_request, "simple request");
}