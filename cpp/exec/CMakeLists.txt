add_executable(hx-exec main.cpp)
target_include_directories(hx-exec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hx-exec PRIVATE hx_common)

add_executable(hx-exec-tests tests.cpp)
target_include_directories(hx-exec-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME hx-exec-tests COMMAND hx-exec-tests)
