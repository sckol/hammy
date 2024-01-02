# ADD SCRIPT THAT WILL MAKE ulimit -s 64M
cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set stack size limit using ulimit
execute_process(
    COMMAND ulimit -s 64M
)

# Add debug flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

#include(CTest)
#enable_testing()

add_executable(hammy simulator/hammy.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
