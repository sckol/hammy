# ADD SCRIPT THAT WILL MAKE ulimit -s 64M
cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
Set(FETCHCONTENT_QUIET FALSE)

# # Fetch Arrow
# FetchContent_Declare(
#   arrow
#   GIT_REPOSITORY https://github.com/apache/arrow.git
#   GIT_TAG apache-arrow-14.0.2
#   GIT_PROGRESS TRUE
# )
# FetchContent_MakeAvailable(arrow)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)


add_executable(hammy_unlimited_stack simulator/hammy_unlimited_stack.cpp)
add_executable(hammy simulator/hammy.cpp)
#target_link_libraries(hammy PRIVATE arrow_shared arrow_bundled_dependencies)
#target_include_directories(hammy PRIVATE ${arrow_SOURCE_DIR}/cpp/src)
add_dependencies(hammy_unlimited_stack hammy)

# Link hammy with Arrow
target_link_libraries(hammy PRIVATE Arrow::arrow_shared Parquet::parquet_shared)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
