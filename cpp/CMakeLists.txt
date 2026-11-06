cmake_minimum_required(VERSION 3.16)
project(gaintrie CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(gaintrie_lib STATIC
  src/code_utils.cpp
  src/gain_coder.cpp
  src/pair_writer.cpp
  src/matrix_coder.cpp
  src/json.cpp
  src/sha256.cpp
  src/pipeline.cpp)
target_include_directories(gaintrie_lib PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(gaintrie_lib PUBLIC Threads::Threads)
# float32 entropies must round like numpy (split tie-breaking): no FMA contraction
target_compile_options(gaintrie_lib PRIVATE -Wall -Wextra -ffp-contract=off)

add_executable(gaintrie src/main.cpp)
target_link_libraries(gaintrie PRIVATE gaintrie_lib)

enable_testing()
add_executable(test_roundtrip tests/test_roundtrip.cpp)
target_link_libraries(test_roundtrip PRIVATE gaintrie_lib)
add_test(NAME roundtrip COMMAND test_roundtrip)
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE gaintrie_lib)
add_test(NAME pipeline COMMAND test_pipeline)
