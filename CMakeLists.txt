cmake_minimum_required(VERSION 3.16)

project(fastcc-benchmark)

#random link

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Abseil requires C++14
set(CMAKE_CXX_STANDARD 20)

add_executable(benchmark benchmark.cpp)
add_executable(chem_benchmark chem_benchmark.cpp)
add_executable(synthetic_benchmark synthetic_benchmark.cpp)

add_subdirectory(abseil-cpp)
add_subdirectory(unordered_dense)
add_subdirectory(taskflow)
add_subdirectory(hopscotch-map)

# Add include directories for Abseil
target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp/absl
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Abseil libraries to benchmark
target_link_libraries(benchmark PRIVATE
    unordered_dense::unordered_dense
    absl::container_memory
    absl::hash_container_defaults
    absl::raw_hash_set
    absl::algorithm_container 
    absl::core_headers
    absl::memory
    absl::type_traits
)

# Add include directories for Abseil
target_include_directories(chem_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp/absl
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Abseil libraries to chem_benchmark
target_link_libraries(chem_benchmark PRIVATE
    unordered_dense::unordered_dense
    absl::container_memory
    absl::hash_container_defaults
    absl::raw_hash_set
    absl::algorithm_container 
    absl::core_headers
    absl::memory
    absl::type_traits
)

# Add include directories for Abseil
target_include_directories(synthetic_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/abseil-cpp/absl
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense
    ${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map
    ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Abseil libraries to chem_benchmark
target_link_libraries(synthetic_benchmark PRIVATE
    unordered_dense::unordered_dense
    absl::container_memory
    absl::hash_container_defaults
    absl::raw_hash_set
    absl::algorithm_container 
    absl::core_headers
    absl::memory
    absl::type_traits
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -g -fopenmp -lstdc++")
