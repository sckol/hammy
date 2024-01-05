cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

file(GLOB JINJA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/queries/*.jinja2)
add_custom_target(process_jinja DEPENDS ${JINJA_FILES}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/process_jinja.py ${CMAKE_CURRENT_SOURCE_DIR}/queries
)


add_executable(hammy simulator/hammy.cpp)
target_link_libraries(hammy PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
add_dependencies(hammy process_jinja)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

