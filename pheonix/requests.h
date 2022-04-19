
#include <iostream>
#include <scheduler.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <curl/curl.h>

/* TODO
* 0) fix aws segfault
* 1) turn into storage provider abstractions
* 2) optimize data buffer transfer
*/

std::string fetch_file(std::string path) 
{
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return std::move(buffer.str());
}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string fetch_http(std::string url) 
{
  CURL *curl;
  CURLcode res; 
  std::string readBuffer;
  auto URL = url.c_str();

  curl = curl_easy_init();

  if (curl){
    curl_easy_setopt(curl, CURLOPT_URL, URL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl); //we really want to make this awaitable
    curl_easy_cleanup(curl);
  }
  return std::move(readBuffer);
}

std::string _fetch_s3(std::string uri) {
  //get credentials
  const Aws::String region = "us-east-1";
  std::cout << "54" << std::endl;
  //FIXME this should be initialized only once
  Aws::Client::ClientConfiguration config;
  // if (!region.empty())
  //    config.region = region;
  Aws::S3::S3Client s3_client(config);
  
  //split uri into bucket_name and key
  std::cout << "62" << std::endl;
  auto uri_shortened = uri.substr(5);
  Aws::String bucket_name = uri_shortened.substr(0, uri_shortened.find("/"));
  Aws::String key = uri_shortened.substr(uri_shortened.find("/")+1);
  std::cout << bucket_name + "72" + key << std::endl;

  // get the object from s3
  Aws::S3::Model::GetObjectRequest object_request;
  object_request.SetBucket(bucket_name);
  std::cout << bucket_name << std::endl;
  std::cout << key << std::endl;
  //object_request.SetKey(key);
  std::cout << __LINE__ << std::endl;
  Aws::S3::Model::GetObjectOutcome get_object_outcome = s3_client.GetObject(object_request);
  
  std::cout << "72" << std::endl;
  // if the object was fetched successfully
  if (get_object_outcome.IsSuccess())
  {
    std::stringstream buffer;
    buffer << get_object_outcome.GetResultWithOwnership().GetBody().rdbuf();
    return std::move(buffer.str()); 
  }
  else
  {
    auto err = get_object_outcome.GetError();
    std::cout << "Error: GetObject: " <<
        err.GetExceptionName() << ": " << err.GetMessage() << std::endl;
    return "";
  }
}

std::string fetch_s3(std::string path){
  std::string response;
  Aws::SDKOptions options;
  options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
  std::cout << "93" << std::endl;
  Aws::InitAPI(options);
  {
    std::cout << "96" << std::endl;
    response = _fetch_s3(path); 
  }
  std::cout << "99" << std::endl;
  Aws::ShutdownAPI(options);
  return std::move(response);
}

auto fetch(std::string path)
{
  if (path.substr(0, 4).compare("http") == 0) {
    return fetch_http(path);
  } 

  if (path.substr(0, 7).compare("file://") == 0) {
    return fetch_file(path.substr(7));
  }

  if (path.substr(0, 5).compare("s3://") == 0) {
    return fetch_s3(path);
  }

  if (path.substr(0, 5).compare("gs://") == 0) {
    //implement google cloud sdk
  }
  std::cout << "Error: no storage provider found from path " << path  << std::endl;
  return path;
}

// prefetch Awaitable
template <typename T> struct prefetch_Awaitable {
  T value;
  std::string readBuffer;

  prefetch_Awaitable(T value) : value(value) {}

  bool await_ready() { return false; }
  std::string &await_resume() { return readBuffer; }

  template <typename Handle> auto await_suspend(Handle h) {
    readBuffer = fetch(value);
    auto &q = scheduler;
    q.push_back(h);
    return q.pop_front();
  }
};

template <typename T> auto prefetch(T value) {
  return prefetch_Awaitable<T>{value};
}


