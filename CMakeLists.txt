cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(comp_dyn_rewrite)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add your project files

file(GLOB H_FILES ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} main.cpp ${H_FILES} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::glfw)
