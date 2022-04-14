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
  auto URL = "http://localhost:8000";
  //auto URL = "http://google.com";
  if (curl){
    curl_easy_setopt(curl, CURLOPT_URL, URL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl); //we really want to make this awaitable
    curl_easy_cleanup(curl);
  }

  // std::cout << readBuffer << std::endl;

  //co_return py::bytes(readBuffer);
  return std::move(readBuffer); //py::bytes()
}

// prefetch Awaitable
template <typename T> struct prefetch_Awaitable {
  T value;
  std::string readBuffer;

  prefetch_Awaitable(T value) : value(value) {}

  bool await_ready() { return false; }
  std::string &await_resume() { return readBuffer; }

  template <typename Handle> auto await_suspend(Handle h) {
    // std::cout << value << std::endl;
    readBuffer = simple_request(value);
    auto &q = scheduler;
    q.push_back(h);
    return q.pop_front();
  }
};

template <typename T> auto prefetch(T value) {
  return prefetch_Awaitable<T>{value};
}
