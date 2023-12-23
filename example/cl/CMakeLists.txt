cmake_minimum_required(VERSION 3.2)

# Define project name
project(ptk_cl)
set(PROJECT_NAME ptk_cl)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# Release by default
set(CMAKE_BUILD_TYPE "Release")  # Debug / Release

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Set for .so
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g") 
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Include
SET(INCLUDE_DIR
    ${OPENCL_WIN_SDK_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
message(STATUS "include: ${INCLUDE_DIR}")
include_directories(${INCLUDE_DIR})
	
# Source GLOB_RECURSE
file(GLOB SRC_LIST 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" # PROJECT_SOURCE_DIR
    # Include
    # "${INCLUDE_DIR}/*.h"
)
message(STATUS "src: ${SRC_LIST}")

# add_library(${PROJECT_NAME} STATIC  ${SRC_LIST})  #SHARED
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OPENCL_WIN_SDK_LIB}/OpenCL.lib)

# Set output path
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})