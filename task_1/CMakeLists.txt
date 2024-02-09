cmake_minimum_required(VERSION 3.0.0)
project(task1 VERSION 0.1.0 LANGUAGES C CXX)


set(SOURCES task1.cpp)

add_executable(float ${SOURCES})
target_compile_definitions(float PUBLIC FLOAT_TYPE)

add_executable(double ${SOURCES})