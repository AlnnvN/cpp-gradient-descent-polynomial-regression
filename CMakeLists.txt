cmake_minimum_required(VERSION 3.16...3.25)

if(${CMAKE_VERSION} VERSION_LESS 3.16)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(VERSION 3.16)
endif()

project(gradient_descent VERSION 1.0
                DESCRIPTION ""
                LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
set(CMAKE_BUILD_TYPE Debug)

add_library(gradient_descent gradient_descent.hpp gradient_descent.cpp)

add_executable(demo demo.cpp)
target_link_libraries(demo LINK_PUBLIC gradient_descent)
                
