cmake_minimum_required(VERSION 3.18)
project(nvenc_uncapped LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
if(NOT EXISTS "${SDK_TOP}/Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp")
    message(FATAL_ERROR "SDK_TOP must point to the installed Video Codec SDK 13 directory")
endif()
add_executable(nvenc_uncapped nvenc_uncapped.cpp
    "${SDK_TOP}/Samples/NvCodec/NvEncoder/NvEncoder.cpp"
    "${SDK_TOP}/Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp")
target_include_directories(nvenc_uncapped PRIVATE
    "${SDK_TOP}/Samples" "${SDK_TOP}/Samples/NvCodec"
    "${SDK_TOP}/Samples/Utils" "${SDK_TOP}/Interface")
find_library(SATC_NVENC_LIBRARY NAMES nvidia-encode libnvidia-encode.so.1 REQUIRED)
target_link_libraries(nvenc_uncapped PRIVATE CUDA::cuda_driver ${SATC_NVENC_LIBRARY} pthread dl)
