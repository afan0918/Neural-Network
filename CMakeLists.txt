cmake_minimum_required(VERSION 3.22)
project(nn)

set(CMAKE_CXX_STANDARD 17)

add_executable(nn main.c
        neural_network.c
        neural_network.h)

target_link_libraries(nn m)