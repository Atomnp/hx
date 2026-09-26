cmake_minimum_required(VERSION 3.20)
project(hx_native LANGUAGES CXX)

# Native helpers for the hx harness. Each subdirectory builds one standalone binary
# that the Python harness runs as a subprocess.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile_commands.json, for editors / clangd

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)
enable_testing()

add_subdirectory(search)
