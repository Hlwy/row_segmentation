cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wno-write-strings")
set(CMAKE_CXX_FLAGS "  ${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")
if(VERBOSE)
message(STATUS "------ BUILDING SOURCE FILES")
endif(VERBOSE)

include_directories(${INC_DIR}/rtl)
add_library(colorfilter SHARED filters/color_filter.cpp)
target_link_libraries(colorfilter ${LIBS})

add_library(filterpipeline SHARED filters/filter_pipeline.cpp)
target_link_libraries(filterpipeline colorfilter ${LIBS})

add_library(ransaclinefitter SHARED algorithms/ransaclinefitter.cpp)
target_link_libraries(ransaclinefitter ${LIBS})

add_executable(process_image process_image.cpp)
target_link_libraries(process_image colorfilter filterpipeline ransaclinefitter ${LIBS})

# include_directories(${INCS_DIR}/actuators actuators)

# # DC Motor
# add_library(dcmotor SHARED dc_motor.cpp)
# target_link_libraries(dcmotor utils ${LIBS})
#
# # Servo
# add_library(servo SHARED servo.cpp)
# target_link_libraries(servo i2c ${LIBS})
#
# if(BUILD_TESTS)
#
#      if(VERBOSE)
#      message(STATUS "------------ Building tests")
#      endif(VERBOSE)
#
#      # DC Motor
#      include_directories(${TEST_DIR}/actuators)
#      add_executable(test_dcmotor ${TEST_DIR}/actuators/test_dc.cpp)
#      target_link_libraries(test_dcmotor dcmotor)
#
#      # Servo
#      add_executable(test_servo ${TEST_DIR}/actuators/test_servo.cpp)
#      target_link_libraries(test_servo servo)
#
# endif(BUILD_TESTS)
