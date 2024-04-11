cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(Protobuf REQUIRED)

file(GLOB JINJA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/queries/*.jinja2)
add_custom_target(process_jinja DEPENDS ${JINJA_FILES}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/process_jinja.py ${CMAKE_CURRENT_SOURCE_DIR}/queries
)
 

add_executable(hammy simulator/hammy.cpp)
target_link_libraries(hammy PRIVATE Arrow::arrow_static Parquet::parquet_static)
add_dependencies(hammy process_jinja)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(cuda_test_cu cuda_test.cu)
target_compile_options(cuda_test_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
set_target_properties(cuda_test_cu PROPERTIES CUDA_ARCHITECTURES "70")

