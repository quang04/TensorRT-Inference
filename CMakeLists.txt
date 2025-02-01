cmake_minimum_required(VERSION 3.16)

project(App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick)

# CUDA
# currently using 11.4
message(STATUS "CUDA SETUP")

if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")
    else()
    endif()
endif()

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
message(STATUS "CUDA SETUP Window")
set(CUDA_LIB
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppc.lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppicc.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
message(STATUS "CUDA SETUP Linux")
set(CUDA_LIB
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicc.so)
else()
message(STATUS "CUDA SETUP Not support")
endif()


# OPENCV
include_directories(${CMAKE_SOURCE_DIR}/external/opencv/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
message(STATUS "OPENCV SETUP Window")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENCV_LIB ${CMAKE_SOURCE_DIR}/external/opencv/lib/opencv_world4100d.lib)
else()
    set(OPENCV_LIB ${CMAKE_SOURCE_DIR}/external/opencv/lib/opencv_world4100.lib)
endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
message(STATUS "OPENCV SETUP Linux")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENCV_LIB ${CMAKE_SOURCE_DIR}/external/opencv/lib/debug/libopencv_world.so)
else()
    set(OPENCV_LIB ${CMAKE_SOURCE_DIR}/external/opencv/lib/release/libopencv_world.so)
endif()
else()
message(STATUS "OPENCV SETUP Not support")
endif()

# ONNX
include_directories(${CMAKE_SOURCE_DIR}/external/onnx/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
message(STATUS "ONNX SETUP Window")
set(ONNX_LIB
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/onnxruntime.lib
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/onnxruntime_providers_shared.lib
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/onnxruntime_providers_cuda.lib
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/onnxruntime_providers_tensorrt.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
message(STATUS "ONNX SETUP Linux")
set(ONNX_LIB
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/libonnxruntime.so
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/libonnxruntime_providers_shared.so
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/libonnxruntime_providers_cuda.so
    ${CMAKE_SOURCE_DIR}/external/onnx/lib/libonnxruntime_providers_tensorrt.so)
else()
message(STATUS "ONNX SETUP Not support")
endif()


# TENSORRT
include_directories(${CMAKE_SOURCE_DIR}/external/trt/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
message(STATUS "TENSORRT SETUP Window")
set(TENSOR_RT_LIB
    ${CMAKE_SOURCE_DIR}/external/trt/lib/nvinfer.lib
    ${CMAKE_SOURCE_DIR}/external/trt/lib/nvinfer_plugin.lib
    ${CMAKE_SOURCE_DIR}/external/trt/lib/nvonnxparser.lib
    ${CMAKE_SOURCE_DIR}/external/trt/lib/nvparsers.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
message(STATUS "TENSORRT SETUP Linux")
set(TENSOR_RT_LIB
    ${CMAKE_SOURCE_DIR}/external/trt/lib/libnvinfer.so
    ${CMAKE_SOURCE_DIR}/external/trt/lib/libnvinfer_plugin.so
    ${CMAKE_SOURCE_DIR}/external/trt/lib/libnvonnxparser.so
    ${CMAKE_SOURCE_DIR}/external/trt/lib/libnvparsers.so)
else()
endif()


qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(appML
    main.cpp
)

qt_add_qml_module(appML
    URI App
    VERSION 1.0
    QML_FILES Main.qml
    QML_FILES App.qml
    SOURCES appvm.h appvm.cpp
    SOURCES inferencethread.h inferencethread.cpp
    SOURCES appdef.h
    SOURCES inferenceservice.h
    SOURCES threadservice.h threadservice.cpp
    SOURCES tensorRTLogger.h
    SOURCES utility.h utility.cpp
)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appML PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appApp
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_link_libraries(appML
    PRIVATE Qt6::Quick
    ${OPENCV_LIB}
    ${ONNX_LIB}
    ${CUDA_LIB}
    ${TENSOR_RT_LIB}
)

include(GNUInstallDirs)
install(TARGETS appML
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
