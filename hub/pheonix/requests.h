#include <curl/curl.h>
#include <iostream>
#include <scheduler.h>


static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

auto simple_request(int a) {
  CURL *curl;
  CURLcode res;
  std::string readBuffer;

  curl = curl_easy_init();
  if (curl){
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl); //we really want to make this awaitable
    curl_easy_cleanup(curl);
  }
  std::cout << a << std::endl;
  //co_return py::bytes(readBuffer);
  return py::bytes(readBuffer);
}

// prefetch Awaitable
template <typename T> struct prefetch_Awaitable {
  T &value;

  prefetch_Awaitable(T &value) : value(value) {}

  bool await_ready() { return false; }
  T &await_resume() { return value; }
  template <typename Handle> auto await_suspend(Handle h) {
    simple_request(1);
    auto &q = scheduler;
    q.push_back(h);
    return q.pop_front();
  }
};

template <typename T> auto prefetch(T value) {
  return prefetch_Awaitable<T>{value};
}
