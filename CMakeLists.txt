cmake_minimum_required(VERSION 3.16...3.25)

if(${CMAKE_VERSION} VERSION_LESS 3.16)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(VERSION 3.16)
endif()

project(gradient_descent VERSION 1.0
                DESCRIPTION ""
                LANGUAGES CXX)

add_library(gradient_descent gradient_descent.hpp gradient_descent.cpp)

add_executable(demo demo.cpp)
target_link_libraries(demo LINK_PUBLIC gradient_descent)
                
