cmake_minimum_required(VERSION 3.10)
project(pheonix)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

file(GLOB_RECURSE SOURCES "*.cpp" "*.c")

ADD_LIBRARY(pheonix ${SOURCES})