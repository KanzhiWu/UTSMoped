cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()


find_package(OpenCV REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package( Eigen REQUIRED )
add_custom_target(BUILD_LIBMOPED DEPENDS libmoped.a)
add_custom_command(OUTPUT libmoped.a 
                   DEPENDS ${PROJECT_SOURCE_DIR}/libmoped/src/moped.cpp
                   DEPENDS ${PROJECT_SOURCE_DIR}/libmoped/src/config.hpp
                   DEPENDS ${PROJECT_SOURCE_DIR}/libmoped/src/util.hpp
                   DEPENDS ${PROJECT_SOURCE_DIR}/libmoped/src/STATUS_DISPLAY.hpp
                   DEPENDS ${PROJECT_SOURCE_DIR}/libmoped/src/GLOBAL_DISPLAY.hpp
                   COMMAND make
                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/libmoped
                   COMMENT Building libmoped.a)


rosbuild_add_boost_directories()
link_directories(${PROJECT_SOURCE_DIR}/libmoped/lib)
include_directories(${PROJECT_SOURCE_DIR}/libmoped/include)
include_directories(/usr/include/vtk-5.8)
include_directories( ${EIGEN_INCLUDE_DIR} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

rosbuild_add_executable(uts_moped uts_moped.cpp)
rosbuild_add_compile_flags(uts_moped "-O3 -Wall -march=native -mfpmath=sse -fopenmp")
rosbuild_add_link_flags(uts_moped "-Wl,--start-group")
target_link_libraries(uts_moped 
  gomp GLEW glut IL 
  ${PROJECT_SOURCE_DIR}/libmoped/lib/libmoped.a 
  ${OpenCV_LIBS}
  boost_signals
  libvtkCommon.so libvtkFiltering.so libvtkRendering.so
  ${PCL_LIBRARIES}
  )

add_dependencies(uts_moped BUILD_LIBMOPED)

add_custom_command(TARGET uts_moped PRE_BUILD COMMAND ./find_opencv2.sh WORKING_DIRECTORY ..)



#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

