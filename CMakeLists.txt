cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(mnist CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include(${PROJECT_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
  CONANFILE conanfile.txt
  BASIC_SETUP
  CMAKE_TARGETS # individual targets to link to
  BUILD
    missing 
)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    include/mnist/*.h
)
file(GLOB SRCS
    src/*.cpp
)

add_library(${PROJECT_NAME}
    ${SRCS}
    ${HRDS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    CONAN_PKG::eigen
    tensorflow
)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
)

set(DEMO_COMMON_HDRS
    demo/helpers.h
)

set(DEMO_COMMON_SRCS
    demo/helpers.cpp
)


# app
set(APP_NAME fashio_mnist)

add_executable(${APP_NAME}
    demo/main.cpp
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(${APP_NAME} PUBLIC
    include
    demo
)

target_link_libraries(${APP_NAME} ${PROJECT_NAME})


# test
add_executable(${APP_NAME}_test
    test/test.cpp
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(${APP_NAME}_test PUBLIC
    include
    demo
)

target_link_libraries(${APP_NAME}_test
    ${PROJECT_NAME}
    CONAN_PKG::gtest
)

add_custom_target(test_${APP_NAME}
    $<TARGET_FILE:${APP_NAME}_test>
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)