cmake_minimum_required(VERSION 3.10)

add_subdirectory(googletest EXCLUDE_FROM_ALL)
# ensure gtest include directory is added before any of the
# system-wide include directories that might contain gtest
include_directories(BEFORE ${gtest_SOURCE_DIR}/include)

include(GoogleTest)

enable_testing()

# add_executable(test_opencv_detector_matcher src/test_opencv_detector_matcher.cpp)
# target_link_libraries(test_opencv_detector_matcher gtest gtest_main ${OpenCV_LIBS} dataset_io)
add_executable(test_loop_closure_utils src/test_loop_closure_utils.cpp)
target_link_libraries(test_loop_closure_utils gtest gtest_main ${OpenCV_LIBS} pangolin opengv TBB::tbb ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2_ORBSLAM/lib/libDBoW2.so)

add_executable(test_DBoW2 src/test_DBoW2.cpp)
target_link_libraries(test_DBoW2 gtest gtest_main ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2_ORBSLAM/lib/libDBoW2.so)  
