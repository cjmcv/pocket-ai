# Define project name
project(unit_tests)
set(PROJECT_NAME unit_tests)

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

set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Googletest 1.13 requires C++14
set(GOOGLETEST_VERSION 1.13.0) # C++14
set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest/googletest")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARIES gtest gtest_main)
set(GTEST_MAIN_LIBRARY gtest_main)
set(GTEST_LIBRARY gtest)
add_subdirectory(${GTEST_ROOT})

# Include
SET(INCLUDE_LIST 
    ${PROJECT_SOURCE_DIR}/../
    ${GTEST_INCLUDE_DIR}
)
message(STATUS "tests include: ${INCLUDE_LIST}")
include_directories(${INCLUDE_LIST})

# Source # GLOB_RECURSE
file(GLOB TESTS_SRC_LIST 
    "${CMAKE_CURRENT_LIST_DIR}/tests_main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp"
)
message(STATUS "tests src: ${TESTS_SRC_LIST}")

# Build
add_executable(${PROJECT_NAME} ${TESTS_SRC_LIST})

# Depends.
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARY})

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})