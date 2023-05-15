cmake_minimum_required(VERSION 2.8)
project(image_retrieval)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_CXX_STANDARD 11)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
# set(OpenCV_DIR "your/path/to/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})


add_executable(make_voc make_voc.cpp)
target_link_libraries(make_voc ${OpenCV_LIBS} ${DBoW3_LIBS})

 add_executable(query query.cpp)
 target_link_libraries(query ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(one_hot_gen one_hot_gen.cpp)
target_link_libraries(one_hot_gen ${OpenCV_LIBS} ${DBoW3_LIBS})