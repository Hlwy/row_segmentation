cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(row_segmentation)

# ===========================
#       Set Build Options
# ===========================
set(VERBOSE         ON CACHE INTERNAL "CMake execution debugging")
set(BUILD_UTILS     ON CACHE INTERNAL "Build all the classes useful throughout entire project that help make programming cleaner and quicker")
set(BUILD_TESTS     ON CACHE INTERNAL "Build all test executables for each class")

# ===========================
#        Set Compiler
# ===========================
# if(COMPILE_ARM)
#      set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
#      set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
# endif(COMPILE_ARM)

# ===========================
#    Set CMake Properties
# ===========================
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBS_DIR ${LIBRARY_OUTPUT_PATH})
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCS_DIR "${PROJECT_SOURCE_DIR}/include")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/tests")
set(HOME_DIR "${PROJECT_SOURCE_DIR}")

# ============================================
#       Find & Include Dependencies
# ============================================
# Find Armadillo
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)

# Find Boost Libraries
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads)

# Find GTSAM libraries
find_package(GTSAMCMakeTools)
find_package(GTSAM REQUIRED)

# Find OpenCV libraries
find_package(OpenCV REQUIRED)

# OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  MESSAGE(STATUS "Found OpenMP")
ENDIF()

# Find PCL Libraries
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})

include_directories(
     include
     ${Boost_INCLUDE_DIRS}
     ${pigpio_INCLUDE_DIR}
     ${LAPACK_INCLUDE_DIRS}
     ${BLAS_INCLUDE_DIRS}
     ${ARMADILLO_INCLUDE_DIRS}
     ${GTSAM_INCLUDE_DIR}
     ${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

set(LIBS
     gtsam
     ${Boost_LIBRARIES}
     ${pigpiod_if2_LIBRARY}
     ${ARMADILLO_LIBRARIES}
     ${LAPACK_LIBRARIES}
     ${BLAS_LIBRARIES}
     ${OpenCV_LIBS}
     ${CMAKE_THREAD_LIBS_INIT}
	${PCL_LIBRARIES}
)

# ===========================
#     Build Targets
# ===========================

# add_executable(pcd_write_test_pcd_write.cpp)
# target_link_libraries(pcd_write_test $PCL_COMMON_LIBRARIES $PCL_IO_LIBRARIES)
add_subdirectory(src)


# ===========================
#     Install Targets
# ===========================


# ===========================
#         DEBUG
# ===========================
# this is used when searching for include files e.g. using the FIND_PATH() command.
# MESSAGE( STATUS "SHARED: " ${SHARED} )

# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#      message(STATUS "         included='${dir}'")
# endforeach()

# MESSAGE( STATUS "INCLUDE_DIRS: ")
# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#      message(STATUS "  dir='${dir}'")
# endforeach()

# this is used when searching for libraries e.g. using the FIND_LIBRARY() command.
# MESSAGE( STATUS "CMAKE_LIBRARY_PATH: " ${CMAKE_LIBRARY_PATH} )

# the compiler flags for compiling C sources
# MESSAGE( STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )

# the compiler flags for compiling C++ sources
# MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
