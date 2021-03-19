cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/.third_party")
endif()

project(psnr_hvsm LANGUAGES CXX)

add_subdirectory(psnr_hvsm)
