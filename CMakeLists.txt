cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C)

file(GLOB JINJA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/queries/*.jinja2)
add_custom_target(process_jinja DEPENDS ${JINJA_FILES}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/process_jinja.py ${CMAKE_CURRENT_SOURCE_DIR}/queries
)

include_directories(include)

# https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${curdir}/${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(experiments ${CMAKE_CURRENT_SOURCE_DIR}/simulator)
FOREACH(experiment ${experiments})
  message(STATUS ${experiment})
  add_subdirectory(${experiment})
ENDFOREACH()

SUBDIRLIST(libs ${CMAKE_CURRENT_SOURCE_DIR}/lib)
FOREACH(lib ${libs})
  message(STATUS ${lib})
  add_subdirectory(${lib})
ENDFOREACH()
 
# 
#add_library(PCGBasic simulator/test/pcg_basic.c)
#target_link_libraries(hammy PUBLIC PCGBasic)

#[[add_executable(hammy simulator/hammy.cpp)
add_executable(cuda_test_cu cuda_test.cu)
target_compile_options(cuda_test_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
set_target_properties(cuda_test_cu PROPERTIES CUDA_ARCHITECTURES "70")
]]