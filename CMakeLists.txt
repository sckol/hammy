cmake_minimum_required(VERSION 3.0.0)
project(hammy VERSION 0.1.0 LANGUAGES C)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/hammy-lib)
include_directories(${CMAKE_SOURCE_DIR}/hammy-lib/c-libs)

# Function to get subdirectories
function(get_subdirs result dir)
    file(GLOB children LIST_DIRECTORIES true RELATIVE ${dir} ${dir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${dir}/${child})
            list(APPEND dirlist ${dir}/${child})
        endif()
    endforeach()
    set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

# Process libraries
get_subdirs(libs ${CMAKE_CURRENT_SOURCE_DIR}/hammy-lib/c-libs)
if(libs)
    foreach(lib ${libs})
        message(STATUS "Processing library: ${lib}")
        if(EXISTS ${lib}/CMakeLists.txt)
            add_subdirectory(${lib})            
        else()
            message(WARNING "No CMakeLists.txt found in library directory: ${lib}")
        endif()
    endforeach()
else()
    message(WARNING "No library directories found")
endif()

# Process experiments
get_subdirs(experiments ${CMAKE_CURRENT_SOURCE_DIR}/experiments)
if(experiments)
    foreach(experiment ${experiments})
        message(STATUS "Processing experiment: ${experiment}")
        # Get the directory name only
        get_filename_component(exp_name ${experiment} NAME)
        # Extract name after underscore
        string(FIND ${exp_name} "_" underscore_pos)
        if(underscore_pos GREATER -1)
            math(EXPR after_underscore "${underscore_pos} + 1")
            string(SUBSTRING ${exp_name} ${after_underscore} -1 base_name)
            if(EXISTS ${experiment}/${base_name}.c)
                add_executable(${base_name} ${experiment}/${base_name}.c)
                target_link_libraries(${base_name} PRIVATE pcg_basic)
                message(STATUS "Added executable: ${base_name}")
            else()
                message(WARNING "Source file not found: ${experiment}/${base_name}.c")
            endif()
        else()
            message(WARNING "No underscore found in experiment name: ${exp_name}")
        endif()
    endforeach()
else()
    message(WARNING "No experiment directories found")
endif()



# Optional CUDA support
if(CMAKE_CUDA_COMPILER)
    add_executable(cuda_test_cu cuda_test.cu)
    target_compile_options(cuda_test_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
    set_target_properties(cuda_test_cu PROPERTIES CUDA_ARCHITECTURES "70")
endif()