FetchContent_MakeAvailable(Catch2)

add_library(test-base INTERFACE)
target_compile_features(test-base INTERFACE cxx_std_23)
target_link_libraries(test-base INTERFACE Catch2::Catch2WithMain)
target_compile_options(test-base INTERFACE ${WARNING_FLAGS})
target_include_directories(test-base INTERFACE ${CMAKE_SOURCE_DIR})

add_executable(heap-test heap.cpp)
target_link_libraries(heap-test PRIVATE test-base)

include(Catch)
catch_discover_tests(heap-test)
