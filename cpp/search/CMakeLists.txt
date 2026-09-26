# The matching logic lives in a small library so both the binary and the tests link it.
add_library(hx_search_core STATIC glob.cpp ignore.cpp literal.cpp)
target_include_directories(hx_search_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(hx-search main.cpp)
target_link_libraries(hx-search PRIVATE hx_search_core Threads::Threads)

add_executable(hx-search-tests tests.cpp)
target_link_libraries(hx-search-tests PRIVATE hx_search_core)
add_test(NAME hx-search-tests COMMAND hx-search-tests)
